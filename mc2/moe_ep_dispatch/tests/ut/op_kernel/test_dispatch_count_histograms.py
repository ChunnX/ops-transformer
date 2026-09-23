# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU algorithm model, not a CANN build or proof of device DMA/atomic ordering."""

from collections import Counter
import random
import unittest


SENDERS = 32
GROUP_ELEMENTS = 32 * 1024 // 2
HISTOGRAM_BINS = 256


def tile_tokens(bs, cores, topk, metadata_bytes=512):
    return min(max(bs // cores, 1), 4096 // topk, 16384 // metadata_bytes)


def tile_ranges(bs, cores, topk, metadata_bytes=512):
    width = tile_tokens(bs, cores, topk, metadata_bytes)
    return [(begin, min(begin + width, bs)) for begin in range(0, bs, width)]


def histogram_page(values, base, bins):
    """Two b16 registers -> packed b8 data/mask -> BIN0/BIN1 frequency counts."""
    result = [0] * bins
    for offset in range(0, len(values), 256):
        packed = [0xFF] * 256
        packed_mask = [False] * 256
        for half in (0, 1):
            for lane in range(128):
                element = offset + half * 128 + lane
                # Deliberately poison lanes outside the logical tail.
                value = values[element] if element < len(values) else base
                shifted = (value - base) & 0xFFFF
                packed[half * 128 + lane] = shifted & 0xFF
                packed_mask[half * 128 + lane] = element < len(values) and shifted < bins
        for value, valid in zip(packed, packed_mask):
            if valid:
                result[value] = (result[value] + 1) & 0xFFFF
    return result


def histogram(values, scope):
    result = []
    for base in range(0, scope, HISTOGRAM_BINS):
        result.extend(histogram_page(values, base, min(scope - base, HISTOGRAM_BINS)))
    return result


def selected_ranks(experts, experts_per_rank, cached=None):
    ranks = [expert // experts_per_rank if expert >= 0 else -1 for expert in experts]
    if cached is not None:
        return [rank if slot >= 0 else -1 for rank, slot in zip(ranks, cached)]
    suppressed = [rank < 0 for rank in ranks]
    for k in range(len(ranks) - 1):  # Last column cannot suppress anything later.
        for column in range(k + 1, len(ranks)):
            suppressed[column] |= ranks[column] == ranks[k]
    return [rank if not drop else -1 for rank, drop in zip(ranks, suppressed)]


def count_contributions(routes, world, expert_count, aivs, cached=None):
    topk = len(routes[0])
    tiles = tile_ranges(len(routes), aivs, topk)
    rank_info = [None] * len(routes)
    contributions = []
    for core in range(aivs):
        for tile in range(core, len(tiles), aivs):
            begin, end = tiles[tile]
            selected = []
            for token in range(begin, end):
                slots = None if cached is None else cached[token]
                row = selected_ranks(routes[token], expert_count // world, slots)
                rank_info[token] = row
                selected.extend(row)
            flat = [value for row in routes[begin:end] for value in row]
            experts = Counter(dict(enumerate(histogram(flat, expert_count))))
            ranks = Counter(dict(enumerate(histogram(selected, world))))
            contributions.append((core, tile, ranks, experts))
    return rank_info, contributions


def merge_contributions(contributions, tiles, world, expert_count):
    rows = [[0] * world for _ in range(tiles)]
    per_rank, per_expert = [0] * world, [0] * expert_count
    for _, tile, ranks, experts in contributions:
        for rank, count in ranks.items():
            rows[tile][rank] += count
            per_rank[rank] += count
        for expert, count in experts.items():
            per_expert[expert] += count
    return rows, per_rank, per_expert


def oracle(routes, world, expert_count, aivs):
    tiles = tile_ranges(len(routes), aivs, len(routes[0]))
    rows = [[0] * world for _ in tiles]
    per_rank, per_expert = [0] * world, [0] * expert_count
    rank_info = []
    for token in routes:
        seen, selected = set(), []
        for expert in token:
            rank = expert // (expert_count // world) if expert >= 0 else -1
            selected.append(rank if rank >= 0 and rank not in seen else -1)
            if rank >= 0:
                seen.add(rank)
                per_expert[expert] += 1
        rank_info.append(selected)
        for rank in seen:
            per_rank[rank] += 1
    for tile, (begin, end) in enumerate(tiles):
        for row in rank_info[begin:end]:
            for rank in row:
                if rank >= 0:
                    rows[tile][rank] += 1
    return rank_info, (rows, per_rank, per_expert)


def cached_oracle(routes, cached, world, expert_count, aivs):
    """Branch-level oracle: slot validity selects routes; do not perform uncached rank dedup."""
    tiles = tile_ranges(len(routes), aivs, len(routes[0]))
    rows = [[0] * world for _ in tiles]
    per_rank, per_expert = [0] * world, [0] * expert_count
    selected = []
    for token, slots in zip(routes, cached):
        row = []
        for expert, slot in zip(token, slots):
            if expert >= 0:
                per_expert[expert] += 1
            rank = expert // (expert_count // world) if expert >= 0 and slot >= 0 else -1
            row.append(rank)
            if rank >= 0:
                per_rank[rank] += 1
        selected.append(row)
    for tile, (begin, end) in enumerate(tiles):
        for row in selected[begin:end]:
            for rank in row:
                if rank >= 0:
                    rows[tile][rank] += 1
    return selected, (rows, per_rank, per_expert)


class DispatchCountHistogramTest(unittest.TestCase):
    def test_histogram_pages_masks_and_tails(self):
        rng = random.Random(73)
        for scope in (2, 8, 64, 127, 128, 129, 255, 256, 257, 512, 1024, 2048):
            for count in (0, 1, 6, 127, 128, 129, 255, 256, 257, 1026):
                values = [rng.choice((-1, 0, 127, 128, 255, 256, scope - 1, scope)) for _ in range(count)]
                reference = Counter(value for value in values if 0 <= value < scope)
                self.assertEqual(histogram(values, scope), [reference[i] for i in range(scope)])

    def test_u16_histogram_flush_prevents_overflow(self):
        for expert in (0, 127, 128, 255, 256, 2047):
            total = 5 * GROUP_ELEMENTS + 17
            accumulated = 0
            for start in range(0, total, GROUP_ELEMENTS):
                values = [expert] * min(GROUP_ELEMENTS, total - start)
                page = histogram_page(values, expert // 256 * 256, 256)
                accumulated += page[expert % 256]
            self.assertEqual(accumulated, total)

    def test_ub_extent_for_full_register_tail_loads(self):
        for topk in range(1, 33):
            group_tokens = GROUP_ELEMENTS // topk
            group_bytes = ((group_tokens * topk * 2 + 255) // 256) * 256
            for tokens in (1, min(127, group_tokens), min(129, group_tokens), group_tokens):
                elements = tokens * topk
                read_bytes = ((elements + 255) // 256) * 256 * 2
                rank_bytes = ((elements * 2 + 255) // 256) * 256
                self.assertLessEqual(read_bytes, group_bytes)  # Includes the masked second b16 register.
                self.assertLessEqual(2 * rank_bytes, 2 * group_bytes)  # Rank + dedup scratch.
                self.assertLessEqual(256 * 2 + 256 * 4, 2 * group_bytes)  # Narrow + wide histogram.

    def test_all_core_tile_ownership_and_ub_limits(self):
        for bs in (1, 31, 32, 33, 55, 56, 57, 127, 4096, 16387):
            for aivs in (33, 40, 48, 56, 64, 80, 100):
                for topk in (1, 6, 32):
                    for metadata in (512, 1024):
                        tiles = tile_ranges(bs, aivs, topk, metadata)
                        visits = [0] * bs
                        active = set()
                        for core in range(aivs):
                            for tile in range(core, len(tiles), aivs):
                                begin, end = tiles[tile]
                                active.add(core)
                                self.assertLessEqual((end - begin) * topk, 4096)
                                self.assertLessEqual((end - begin) * metadata, 16384)
                                for token in range(begin, end):
                                    visits[token] += 1
                        self.assertEqual(visits, [1] * bs)
                        self.assertEqual(len(active), min(bs, aivs))

    def check_case(self, bs, topk, world, expert_count, aivs, rng):
        routes = [[rng.randrange(-1, expert_count) for _ in range(topk)] for _ in range(bs)]
        expected_info, expected = oracle(routes, world, expert_count, aivs)
        rank_info, contributions = count_contributions(routes, world, expert_count, aivs)
        self.assertEqual(rank_info, expected_info)
        rng.shuffle(contributions)  # Tile completion order must not change counts.
        tiles = tile_ranges(bs, aivs, topk)
        actual = merge_contributions(contributions, len(tiles), world, expert_count)
        self.assertEqual(actual, expected)
        # Tile prefixes retain the original BS-ordered, peer-contiguous source-token lists.
        for rank in range(world):
            indices, prefix = [], 0
            for tile, counts in enumerate(actual[0]):
                begin, end = tiles[tile]
                part = [token for token in range(begin, end) if rank in rank_info[token]]
                self.assertEqual(len(part), counts[rank])
                self.assertEqual(prefix, len(indices))
                indices.extend(part)
                prefix += counts[rank]
            self.assertEqual(indices, [t for t, row in enumerate(expected_info) if rank in row])
        cached = [[0 if rank >= 0 else -1 for rank in row] for row in expected_info]
        cached_info, cached_parts = count_contributions(routes, world, expert_count, aivs, cached)
        self.assertEqual(cached_info, rank_info)
        self.assertEqual(merge_contributions(cached_parts, len(tiles), world, expert_count), expected)

    def test_counts_cached_and_slot_prefix_consistency(self):
        rng = random.Random(4096)
        for aivs in (33, 48, 56, 64, 100):
            for bs, topk, world, experts in (
                    (1, 1, 2, 2), (31, 6, 8, 256), (57, 32, 3, 96),
                    (129, 7, 33, 264), (17, 6, 256, 512), (3, 32, 1024, 2048)):
                self.check_case(bs, topk, world, experts, aivs, rng)
        self.check_case(4096, 6, 8, 256, 56, rng)

    def test_duplicate_experts_same_rank_and_all_invalid(self):
        for routes in ([[-1] * 6] * 65, [[0, 0, 1, 2, 32, -1]] * 65, [[255] * 6] * 65):
            expected_info, expected = oracle(routes, 8, 256, 56)
            info, parts = count_contributions(routes, 8, 256, 56)
            self.assertEqual(info, expected_info)
            self.assertEqual(merge_contributions(parts, len(expected[0]), 8, 256), expected)

    def test_cached_slot_selection_independent_of_uncached_dedup(self):
        # Deliberately include repeated valid same-rank slots to test the cached branch, not cache generation.
        routes = [[0, 1, 31, 32, 95, -1], [64, 65, 66, 33, 34, 35]] * 33
        for aivs in (33, 48, 56, 100):
            for selection in ([7, 9, -1, 0, -1, -1], [-1] * 6, [11] * 5 + [-1]):
                cached = [selection[:] for _ in routes]
                expected_info, expected = cached_oracle(routes, cached, 3, 96, aivs)
                info, parts = count_contributions(routes, 3, 96, aivs, cached)
                self.assertEqual(info, expected_info)
                self.assertEqual(merge_contributions(parts, len(expected[0]), 3, 96), expected)
                self.assertEqual(sum(expected[2]), sum(expert >= 0 for row in routes for expert in row))

    def test_multiple_groups_preserve_expert_accumulation(self):
        # Exercise the per-core chunk boundary and a total exceeding uint16 histogram capacity.
        routes = [[0] * 32] * 18000
        info, parts = count_contributions(routes, 2, 32, 33)
        rows, ranks, experts = merge_contributions(parts, len(tile_ranges(18000, 33, 32)), 2, 32)
        self.assertEqual(sum(row[0] for row in rows), 18000)
        self.assertEqual(ranks, [18000, 0])
        self.assertEqual(experts, [18000 * 32] + [0] * 31)
        self.assertTrue(all(row == [0] + [-1] * 31 for row in info))


if __name__ == "__main__":
    unittest.main()
