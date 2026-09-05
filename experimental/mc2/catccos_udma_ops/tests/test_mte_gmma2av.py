"""CPU model checks for target-order MTE grouping and completion contracts.

These tests compare the consumer's index arithmetic with independently packed
producer rows; they do not substitute for NPU accuracy or memory-order tests.
"""

from collections import Counter
from itertools import permutations

import pytest


def producer_rounds(experts, block_m, groups_per_round):
    """Pack complete M groups in expert order, as the frozen AIC scheduler does."""
    groups = []
    for expert, tokens in enumerate(experts):
        for start in range(0, tokens, block_m):
            groups.append([(expert, row) for row in range(start, min(start + block_m, tokens))])
    return [
        [row for group in groups[start:start + groups_per_round] for row in group]
        for start in range(0, len(groups), groups_per_round)
    ]


def consumer_slice(experts, block_m, groups_per_round, round_index):
    """Closed-form interval intersection, independent of producer packing."""
    first_group = round_index * groups_per_round
    last_group = first_group + groups_per_round
    preceding_rows = actual_rows = preceding_groups = 0
    for tokens in experts:
        group_count = (tokens + block_m - 1) // block_m
        selected_start = min(group_count, max(0, first_group - preceding_groups))
        selected_end = min(group_count, max(0, last_group - preceding_groups))
        row_start = min(tokens, selected_start * block_m)
        row_end = min(tokens, selected_end * block_m)
        preceding_rows += row_start
        actual_rows += row_end - row_start
        preceding_groups += group_count
    return preceding_rows, actual_rows


class IncrementalConsumer:
    """Stateful single-peer scheduler, retaining the preceding round's rows."""

    def __init__(self, experts, block_m, groups_per_round):
        self.experts, self.block_m, self.groups_per_round = experts, block_m, groups_per_round
        self.consumed_rows = self.actual_rows = 0

    def advance(self, round_index):
        if round_index:
            self.consumed_rows += self.actual_rows
        self.actual_rows = 0
        skip = round_index * self.groups_per_round
        remaining = self.groups_per_round
        for tokens in self.experts:
            groups = (tokens + self.block_m - 1) // self.block_m
            if skip >= groups:
                skip -= groups
                continue
            take = min(remaining, groups - skip)
            self.actual_rows += min(take * self.block_m, max(0, tokens - skip * self.block_m))
            remaining -= take
            skip = 0
            if not remaining:
                break
        return self.consumed_rows, self.actual_rows


def assignments(rank, world_size, comm_aivs, workers_per_peer, physical_aics):
    """Enumerate only subcore-0 roles; the self owner is never in the remote pool."""
    result = []
    group_count = comm_aivs // workers_per_peer
    for core in range(physical_aics):
        if core == rank:
            result.append((core, rank, 0, 1))
            continue
        remote_index = core if core < rank else core - 1
        if remote_index >= comm_aivs:
            continue
        group, lane = divmod(remote_index, workers_per_peer)
        for ordinal in range(group, world_size - 1, group_count):
            peer = ordinal if ordinal < rank else ordinal + 1
            result.append((core, peer, lane, workers_per_peer))
    return result


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_default_mapping_preserves_one_original_owner_per_peer(world_size):
    for rank in range(world_size):
        mapped = assignments(rank, world_size, world_size - 1, 1, 24)
        assert sorted(mapped) == [(peer, peer, 0, 1) for peer in range(world_size)]


@pytest.mark.parametrize("groups,workers", [(1, 1), (1, 4), (2, 3), (7, 2), (7, 3)])
def test_remote_groups_cover_every_peer_and_lane_exactly_once(groups, workers):
    world_size = 8
    for rank in range(world_size):
        mapped = assignments(rank, world_size, groups * workers, workers, 24)
        actual = Counter((peer, lane) for _, peer, lane, _ in mapped)
        expected = Counter({(rank, 0): 1})
        expected.update({(peer, lane): 1 for peer in range(world_size) if peer != rank
                         for lane in range(workers)})
        assert actual == expected
        assert all(core != rank for core, peer, _, _ in mapped if peer != rank)


@pytest.mark.parametrize("block_m", [128, 256])
@pytest.mark.parametrize("groups_per_round", [1, 3, 14])
def test_packed_producer_and_stateless_consumer_match_with_expert_tails(block_m, groups_per_round):
    experts = [0, 1, block_m - 1, block_m, block_m + 1, 0, 2 * block_m + 17]
    expected = [(expert, row) for expert, count in enumerate(experts) for row in range(count)]
    rounds = producer_rounds(experts, block_m, groups_per_round)
    output = [None] * len(expected)
    writes = Counter()
    # A worker can revisit a peer after serving another peer; no incremental
    # consumedRows state may be assumed. Deliberately visit rounds out of order.
    for round_index in reversed(range(len(rounds))):
        offset, count = consumer_slice(experts, block_m, groups_per_round, round_index)
        assert count == len(rounds[round_index])
        for index, value in enumerate(rounds[round_index], offset):
            output[index] = value
            writes[index] += 1
    assert output == expected
    assert writes == Counter({index: 1 for index in range(len(expected))})
    assert consumer_slice(experts, block_m, groups_per_round, len(rounds) + 3) == (len(expected), 0)


@pytest.mark.parametrize("block_m", [128, 256])
@pytest.mark.parametrize("groups_per_round", [1, 3, 14])
def test_incremental_single_peer_matches_stateless_and_packed_producer(block_m, groups_per_round):
    routes = [[0] * 8, [1], [0, block_m, 0, block_m + 1],
              [1, 3 * block_m - 1, 0, 2 * block_m + 17, 3]]
    for experts in routes:
        incremental = IncrementalConsumer(experts, block_m, groups_per_round)
        rounds = producer_rounds(experts, block_m, groups_per_round)
        expected_prefix = 0
        for round_index in range(len(rounds) + 3):
            actual = incremental.advance(round_index)
            expected_rows = len(rounds[round_index]) if round_index < len(rounds) else 0
            assert actual == (expected_prefix, expected_rows)
            assert actual == consumer_slice(experts, block_m, groups_per_round, round_index)
            expected_prefix += expected_rows


@pytest.mark.parametrize("group_count", [1, 2, 3, 4, 6, 7])
def test_incremental_path_only_selected_when_group_always_visits_one_peer(group_count):
    remote_peers = 7
    for group in range(group_count):
        visited = list(range(group, remote_peers, group_count))
        sequential_peer = group + group_count >= remote_peers
        assert sequential_peer == (len(visited) == 1)


@pytest.mark.parametrize("workers,chunk_elements", [(1, 17), (2, 1), (4, 31), (8, 65536)])
def test_chunk_workers_reconstruct_output_without_crossing_peer_boundaries(workers, chunk_elements):
    routes = [[0, 1, 129], [255, 0, 3], [0, 0, 0], [257, 13, 128]]
    columns = 5
    expected = [(peer, expert, row, col) for peer, experts in enumerate(routes)
                for expert, tokens in enumerate(experts) for row in range(tokens)
                for col in range(columns)]
    output = [None] * len(expected)
    writes = Counter()
    peer_base = 0
    for peer, experts in enumerate(routes):
        for round_index, packed_rows in enumerate(producer_rounds(experts, 128, 2)):
            rows_before, rows = consumer_slice(experts, 128, 2, round_index)
            payload = [(peer, expert, row, col) for expert, row in packed_rows for col in range(columns)]
            chunk_count = (len(payload) + chunk_elements - 1) // chunk_elements
            for lane in reversed(range(workers)):
                for chunk in range(lane, chunk_count, workers):
                    begin = chunk * chunk_elements
                    limit = min(len(payload), begin + chunk_elements)
                    for index in range(begin, limit):
                        dest = peer_base + rows_before * columns + index
                        assert peer_base <= dest < peer_base + sum(experts) * columns
                        output[dest] = payload[index]
                        writes[dest] += 1
            assert len(payload) == rows * columns
        peer_base += sum(experts) * columns
    assert output == expected
    assert writes == Counter({index: 1 for index in range(len(expected))})


def test_peer_ack_requires_all_workers_including_workers_without_chunks():
    for finish_order in permutations(range(4)):
        done = [0] * 4
        acknowledgements = 0
        for position, lane in enumerate(finish_order):
            done[lane] = 7
            ready = all(generation == 7 for generation in done)
            assert ready == (position == 3)
            if ready:
                acknowledgements += 1
        assert acknowledgements == 1
        # The next communication round must not consume completion from round 7.
        assert not all(generation == 8 for generation in done)
