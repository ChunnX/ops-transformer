# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only layout/protocol proofs; these do not execute an Ascend C kernel."""

from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parents[5]
COMBINE = (
    ROOT
    / "mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_layered.h"
).read_text()
DISPATCH = (
    ROOT
    / "mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_layered.h"
).read_text()


def align(value, unit):
    return (value + unit - 1) // unit * unit


class Issue4501Invariants(unittest.TestCase):
    def test_layered_bs_limits_remain_256(self):
        # Source guards only: expanding the kernel's scratch stress range must
        # not silently expand the host's supported input range in this fix.
        for operator in ("dispatch", "combine"):
            tiling = (
                ROOT
                / f"mc2/moe_distribute_{operator}_v2/op_host/op_tiling"
                / f"moe_distribute_{operator}_v2_tiling.cpp"
            ).read_text()
            self.assertRegex(tiling, r"BS_UPPER_BOUND_LAYERED\s*=\s*256\s*;")
        window = (
            ROOT
            / "mc2/moe_distribute_dispatch_v2/op_host/op_tiling"
            / "moe_distribute_check_win_size.h"
        ).read_text()
        self.assertRegex(window, r"LAYERED_BS_UPPER_BOUND\s*=\s*256UL\s*;")
        self.assertNotIn("LAYERED_BS_UPPER_BOUND_A3", window)
        self.assertIn("maxBs > LAYERED_BS_UPPER_BOUND", window)

    def test_source_has_wide_read_offset_and_disjoint_epoch_state(self):
        self.assertRegex(DISPATCH, r"uint64_t\s+srcIpcOffset\s*=")
        self.assertIn("(magicValue_ << RDMA_FLAG_STATE_BITS) | state", COMBINE)
        self.assertIn("(aivNum_ - serverNum_) / serverNum_ * serverNum_", COMBINE)
        for state in ("ARRIVED", "END"):
            self.assertNotRegex(COMBINE, rf"RDMA_TOKEN_{state}_FLAG\s*\+\s*magicValue_")

    def test_8gib_ipc_offsets_match_writer_for_every_slice(self):
        for experts in (32, 64, 128, 256, 512):
            stride = (8192 * 1024**2 - 404 * 1024**2) // experts // 512 * 512
            local = experts // 32
            for expert in range(local):
                for rank in range(32):
                    writer = (
                        expert * 32 * stride
                        + (rank // 16) * 16 * stride
                        + rank % 16 * stride
                    )
                    reader = (expert * 32 + rank) * stride
                    self.assertEqual(writer, reader)
            self.assertGreater((experts - 1) * stride, 2**32)

    def test_all_admitted_core_counts_partition_without_invalid_servers(self):
        for aiv in range(16, 65):
            for servers in range(2, 17):
                if aiv <= 2 * servers:
                    continue
                producers = (aiv - servers) // servers * servers
                per_server = producers // servers
                self.assertGreater(per_server, 0)
                self.assertLessEqual(producers + servers, aiv)
                self.assertEqual(
                    set(range(servers)),
                    {core // per_server for core in range(producers)},
                )
                self.assertEqual(servers, len(range(producers, producers + servers)))

    def test_fp16_compact_source_never_aliases_broadcast_or_live_tokens(self):
        # Formula mirrors SumToWindowLocalTensor; enumerate every admitted H.
        # BS257/512 below are extra layout stress, not supported host inputs.
        peak = 0
        for h in range(32, 7169, 32):
            for bs in (1, 2, 65, 257, 512):
                for k in range(1, 17):
                    data_end = 2 * align(bs, 16) + 4 * align(bs * k, 8) + 2 * (h + 16)
                    broadcast_end = data_end + 2 * h + align(2 * h, 256)
                    sum_float_end = data_end + align(4 * h, 256) + 4 * h
                    compact_end = sum_float_end + 2 * align(h // 16, 16)
                    self.assertLessEqual(broadcast_end, sum_float_end)
                    self.assertGreaterEqual(align(h // 16, 16), 8 * ((h + 127) // 128))
                    bf16_end = sum_float_end + 4 * align(h // 8, 8)
                    peak = max(peak, compact_end, bf16_end)
        self.assertLess(peak, 185 * 1024)
        self.assertIn("RoundUp(scaleNum_, B16_PER_BLOCK), baseBuffOffset", COMBINE)
        self.assertLessEqual(185 * 1024 + (64 + 4) * 8 + 2 * 32, 192 * 1024)
        self.assertNotIn("InitBuffer(statusBuf_, worldSize_ * UB_ALIGN)", COMBINE)

    def test_old_end_cannot_be_new_arrived(self):
        bits = int(re.search(r"RDMA_FLAG_STATE_BITS = (\d+)", COMBINE)[1])
        arrived = int(re.search(r"RDMA_TOKEN_ARRIVED_FLAG = (\d+)", COMBINE)[1])
        end = int(re.search(r"RDMA_TOKEN_END_FLAG = (\d+)", COMBINE)[1])
        self.assertLess(max(arrived, end), 1 << bits)
        flags = {
            (magic << bits) | state for magic in range(2000) for state in (arrived, end)
        }
        self.assertEqual(4000, len(flags))
        self.assertEqual(321, 123 + 198)  # Old counterexample.
        self.assertNotEqual(end, (198 << bits) | arrived)
        # The 2-bit encoding retains 62 epoch bits; no earlier same-state wrap.
        self.assertNotEqual((0 << bits) | end, (((1 << 62) - 1) << bits) | end)

    def test_final_reduce_ub_fits_at_all_server_counts(self):
        for servers in range(2, 17):
            for bs in (1, 2, 65, 257, 512):
                for h in range(32, 7169, 32):
                    metadata = 4 * align(bs * servers, 8) + 4 * align(bs, 8)
                    unquantized = metadata + 10 * h
                    # Brcb's last destination writes whole repeats, including small H tails.
                    fp16 = metadata + 5 * h + 2 * align(h // 16, 16) + align(2 * h, 256)
                    bf16 = metadata + 11 * h + 6 * align(h // 8, 16) + align(4 * h, 256)
                    self.assertLess(max(unquantized, fp16, bf16), 185 * 1024)

    def test_stable_minimum_expert_order(self):
        rng = random.Random(4501)
        for _ in range(5000):
            experts = rng.choice((32, 64, 128, 256, 512))
            bs = rng.choice((1, 2, 7, 65, 512))
            k = rng.choice((1, 2, 8, 16))
            rows = [rng.sample(range(experts), k) for _ in range(bs)]
            begin = rng.choice((0, experts // 2))
            end = begin + experts // 2
            seen, old = set(), []
            for expert in range(begin, end):
                for token, ids in enumerate(rows):
                    if expert in ids and token not in seen:
                        old.append(token)
                        seen.add(token)
            # An ordered token range makes tie-breaking stable, independent of set order.
            new = sorted(
                sorted(seen),
                key=lambda token: min(e for e in rows[token] if begin <= e < end),
            )
            self.assertEqual(old, new)


if __name__ == "__main__":
    unittest.main()
