# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""CPU route/worker models for MTE dispatch and the unchanged export order.

These exercise route tails and physical chunk partitioning against direct
expert/source-order references; accelerator ordering still needs device tests.
"""
from collections import Counter
from itertools import product
import random

import pytest


def remote_owners(rank, ranks, cores, self_cores, groups, workers):
    result = []
    for core in range(cores):
        extra_self = ranks <= core < ranks + self_cores - 1
        if core == rank or extra_self:
            continue
        index = core if core < rank else core - 1 if core < ranks else core - self_cores
        if index >= groups * workers:
            continue
        group, lane = divmod(index, workers)
        for ordinal in range(group, ranks - 1, groups):
            peer = ordinal if ordinal < rank else ordinal + 1
            result.append((core, peer, lane))
    return result


@pytest.mark.parametrize("ranks,self_cores", [(1, 1), (2, 1), (8, 1), (8, 4), (8, 8)])
def test_remote_workers_never_overlap_fixed_self_or_export_subcores(ranks, self_cores):
    cores = 32
    for rank in range(ranks):
        for groups in range(1, ranks):
            for workers in (1, 2, 4):
                if groups * workers > cores - self_cores:
                    continue
                assignments = remote_owners(rank, ranks, cores, self_cores, groups, workers)
                assert Counter((peer, lane) for _, peer, lane in assignments) == Counter(
                    {(peer, lane): 1 for peer in range(ranks) if peer != rank
                     for lane in range(workers)})
                self_slots = {rank, *range(ranks, ranks + self_cores - 1)}
                assert all(core not in self_slots for core, _, _ in assignments)
                # Remote workers use subcore 1. Export uses subcore 0.
                assert not ({(core, 1) for core, _, _ in assignments}
                            & {(core, 0) for core in range(cores)})
        if ranks > 1:
            assert remote_owners(rank, ranks, cores, self_cores, ranks - 1, 1) == [
                (peer, peer, 0) for peer in range(ranks) if peer != rank]


def chunks(rows, columns, src_stride, dst_stride, chunk_elements, workers):
    """Enumerate bounded physical transfers with per-worker independent rings."""
    flattened = src_stride == dst_stride == columns
    chunk_count = ((rows * columns + chunk_elements - 1) // chunk_elements if flattened else
                   rows * ((columns + chunk_elements - 1) // chunk_elements))
    for lane in range(workers):
        for chunk in range(lane, chunk_count, workers):
            if flattened:
                offset = chunk * chunk_elements
                yield lane, offset, offset, min(chunk_elements, rows * columns - offset)
            else:
                row, column_chunk = divmod(chunk, (columns + chunk_elements - 1) // chunk_elements)
                column = column_chunk * chunk_elements
                yield lane, row * src_stride + column, row * dst_stride + column, min(
                    chunk_elements, columns - column)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("workers,chunk_elements", [(1, 16), (2, 32), (7, 16384)])
def test_copy_tails_preserve_sentinels_and_write_each_element_once(strided, workers, chunk_elements):
    for rows, columns in product((0, 1, 3, 17), (1, 15, 16, 33, 65)):
        src_stride = columns + (3 if strided else 0)
        dst_stride = columns + (5 if strided else 0)
        src = list(range(rows * src_stride))
        dst = [-1] * (rows * dst_stride)
        writes = Counter()
        for _, begin, target, count in chunks(rows, columns, src_stride, dst_stride, chunk_elements, workers):
            assert 0 < count <= chunk_elements
            dst[target:target + count] = src[begin:begin + count]
            writes.update(range(target, target + count))
        expected_positions = {row * dst_stride + col for row in range(rows) for col in range(columns)}
        assert writes == Counter({index: 1 for index in expected_positions})
        for row in range(rows):
            assert dst[row * dst_stride:row * dst_stride + columns] == src[row * src_stride:row * src_stride + columns]
            assert dst[row * dst_stride + columns:(row + 1) * dst_stride] == [-1] * (dst_stride - columns)


@pytest.mark.parametrize("ranks,stage_rows,workers", [(1, 4, 1), (2, 7, 3), (8, 16, 4)])
def test_staged_peer_puts_and_export_equal_expert_source_order(ranks, stage_rows, workers):
    rng = random.Random(842)
    experts = 3
    route = [[[rng.randrange(0, 2 * stage_rows + 2) for _ in range(experts)]
              for _ in range(ranks)] for _ in range(ranks)]
    if ranks > 1:
        route[0][1] = [0] * experts
        route[-1][0] = [1, 0, 3 * stage_rows + 1]
    # Each sender input is destination/expert/token ordered.
    inputs = [[(source, peer, expert, token) for peer in range(ranks)
               for expert, count in enumerate(route[source][peer]) for token in range(count)]
              for source in range(ranks)]
    outputs = []
    for peer in range(ranks):
        expected = [(source, peer, expert, token) for expert in range(experts)
                    for source in range(ranks) for token in range(route[source][peer][expert])]
        output = [None] * len(expected)
        writes = Counter()
        loops = max((sum(route[source][peer]) + stage_rows - 1) // stage_rows for source in range(ranks))
        for round_index in range(loops):
            stage = {}
            for source in range(ranks):
                count = sum(route[source][peer])
                rows = min(stage_rows, max(0, count - round_index * stage_rows))
                base = sum(sum(route[source][p]) for p in range(peer))
                offset = base + round_index * stage_rows
                packed = [None] * rows
                for _, start, dest, size in chunks(rows, 1, 1, 1, 16, workers):
                    packed[dest:dest + size] = inputs[source][offset + start:offset + start + size]
                stage[source] = packed
            # Export intersects each expert's source interval with this stage.
            source_offsets = [0] * ranks
            output_offset = 0
            for expert in range(experts):
                for source in range(ranks):
                    count = route[source][peer][expert]
                    first = source_offsets[source]
                    begin = max(first, round_index * stage_rows)
                    end = min(first + count, (round_index + 1) * stage_rows)
                    if begin < end:
                        start = begin - round_index * stage_rows
                        target = output_offset + begin - first
                        output[target:target + end - begin] = stage[source][start:start + end - begin]
                        writes.update(range(target, target + end - begin))
                    source_offsets[source] += count
                    output_offset += count
        assert output == expected
        assert writes == Counter({index: 1 for index in range(len(expected))})
        outputs.append(output)
    assert sum(map(len, outputs)) == sum(map(len, inputs))


@pytest.mark.parametrize("columns", [1, 255, 256, 257, 2047, 2048])
@pytest.mark.parametrize("stages", [2, 3])
def test_aic_reads_mte_padded_rows_and_capacity_includes_padding(columns, stages):
    ranks, rows = 8, 3
    stride = ((columns + 255) // 256) * 256
    capacity = stages * ranks * rows * stride
    memory = [None] * capacity
    for stage, rank, row in product(range(stages), range(ranks), range(rows)):
        # AIV uses one DistRowMajor allocation per stage.
        base = stage * ranks * rows * stride + rank * rows * stride + row * stride
        for col in range(columns):
            memory[base + col] = (stage, rank, row, col)
    for stage, rank, row in product(range(stages), range(ranks), range(rows)):
        # AIC flattens stage/rank/row but must retain the aligned GM stride.
        flat_row = (stage * ranks + rank) * rows + row
        assert memory[flat_row * stride:flat_row * stride + columns] == [
            (stage, rank, row, col) for col in range(columns)]
    if columns % 256:
        assert capacity > stages * ranks * rows * columns
        assert memory[columns] is None  # Raw-K row addressing would read padding.
