# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU submission model and source contracts, not NPU/SQ-progress or DMA-ordering proof."""

from pathlib import Path
import re
import unittest


MC2 = Path(__file__).resolve().parents[2]
SENDERS = 32
FIRST_BATCH_WQEBB = 32


def read_kernel(op):
    return (MC2 / op / "op_kernel" / f"{op}.h").read_text(encoding="utf-8")


def function_body(source, name):
    start = source.index(f"::{name}(")
    begin = source.index("{", start)
    depth = 1
    for end in range(begin + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin + 1:end]
    raise ValueError(f"Unclosed function: {name}")


class AsyncBatch:
    def __init__(self, capacity, commit_on_full):
        self.capacity = capacity
        self.commit_on_full = commit_on_full
        self.pending = []
        self.pending_wqebb = 0
        self.submitted = []
        self.batch_sizes = []

    def limit(self):
        return self.capacity if self.batch_sizes else FIRST_BATCH_WQEBB

    def append(self, event, wqebb):
        if self.pending_wqebb + wqebb > self.limit():
            self.commit()
        self.pending.append(event)
        self.pending_wqebb += wqebb
        if self.commit_on_full and self.pending_wqebb >= self.limit():
            self.commit()

    def commit(self):
        if self.pending:
            self.submitted.extend(self.pending)
            self.batch_sizes.append(self.pending_wqebb)
            self.pending.clear()
            self.pending_wqebb = 0


class AsyncCompletionTest(unittest.TestCase):
    def test_sender_sources_have_no_benchmark_wait_or_cqe(self):
        for op in ("moe_ep_dispatch", "moe_ep_combine"):
            source = read_kernel(op)
            self.assertNotIn(".Drain(", source)
            self.assertNotIn("WaitIncomingCompletion", source)
            self.assertNotIn("HCOMM_DRAIN_BATCH_WQEBB", source)
            self.assertEqual(re.findall(r"\.cqe\s*=\s*(\d+)", source), ["0", "0"])
            self.assertIn("HCOMM_FIRST_BATCH_WQEBB = 32U", source)

    def test_tail_commits_and_first_batch_reset_are_retained(self):
        cases = (("moe_ep_dispatch", "WriteToRemoteWindow", "SendTokenPipeline();", "CommitBatch(true);"),
                 ("moe_ep_combine", "SendPhaseFromMetadata", "SendMetadataPipeline();", "FlushPreparedWrites();"))
        for op, name, send, flush in cases:
            body = function_body(read_kernel(op), name)
            self.assertLess(body.index("firstBatchCommitted_ = false;"), body.index(send))
            self.assertLess(body.index(send), body.index(flush))
            self.assertLess(body.index(flush), body.rindex("DataCacheCleanAndInvalid"))

    def test_payload_flags_remain_ordered(self):
        for op, payload, flag in (("moe_ep_dispatch", "DATA_CFG", "NOTIFY_CFG"),
                                  ("moe_ep_combine", "DEFAULT_WQE_CONFIG", "JETTY_FLAG_WQE_CONFIG")):
            source = read_kernel(op)
            for config, odr in ((payload, 5), (flag, 6)):
                self.assertRegex(source, rf"{config}\s*=\s*\{{\.odr = {odr}, \.fence = 1")

    def test_dispatch_still_waits_for_tiles_and_receive_counts(self):
        source = read_kernel("moe_ep_dispatch")
        body = function_body(source, "WriteToRemoteWindow")
        self.assertLess(body.index("LoadTileDescriptor(tile);"), body.index("SendTokenPipeline();"))
        self.assertIn("WaitTileReady(tile);", function_body(source, "LoadTileDescriptor"))
        body = function_body(source, "Process")
        self.assertIn("GetRecvCount();", body)
        self.assertLess(body.index("CalSendCnt();"), body.index("Communication();"))
        self.assertLess(body.index("BuildSendPrefixes();"), body.index("Communication();"))
        self.assertLess(body.index("ReduceExpertPartials();"), body.index("Communication();"))
        prepare = function_body(source, "PrepareTile")
        self.assertLess(prepare.index("CopyLocalMetadata("), prepare.index("PublishTileReady(tile);"))

    def test_epilogues_keep_wait_clear_and_consumer_barrier(self):
        source = read_kernel("moe_ep_dispatch_epilogue")
        body = function_body(source, "Process")
        self.assertEqual(body.count("WaitDispatch();"), 2)  # Cached and uncached.
        self.assertRegex(body, r"WaitDispatch\(\);\s*SyncAll<true>\(\);")
        self.assertIn("ClearDispatchFlags();", function_body(source, "WaitDispatch"))
        source = read_kernel("moe_ep_combine_epilogue")
        body = function_body(source, "RecvPhaseReduce")
        self.assertLess(body.index("WaitCombineCompletion()"), body.index("ClearCompletionFlags();"))
        self.assertLess(body.index("ClearCompletionFlags();"), body.index("SyncAll<true>();"))
        self.assertLess(body.index("SyncAll<true>();"), body.index("LoadLocalRecvIndex();"))

    def test_actual_a_remains_optional_and_has_one_publisher(self):
        source = read_kernel("moe_ep_dispatch")
        receive = function_body(source, "GetRecvCount")
        self.assertRegex(receive, r"if \(aivId_ != aivNum_ - 1\)\s*\{[^}]*return;")
        self.assertEqual(receive.count("SetRecvNumPerExpert();"), 1)
        body = function_body(source, "SetRecvNumPerExpert")
        publish = body[body.index("if constexpr (DoCpuSync)"):]
        self.assertLess(publish.index("ReduceSum(recvCntTensor, numRecvPerExpertTensor_"),
                        publish.index("DataCopy(hostPinnedCounterTensor"))
        self.assertRegex(publish, r"DataCopy\(hostPinnedCounterTensor[^;]+;\s*\}")
        self.assertEqual(source.count("DataCopy(hostPinnedCounterTensor"), 1)

    def test_cached_slots_are_copied_before_histogram_scratch_reuse(self):
        body = function_body(read_kernel("moe_ep_dispatch"), "CalSendCntGroup")
        copy = body.index("DataCopyPad(dstSlotIdxGMTensor_[topkIdxOffset], dstSlotIdxTensor_, copyParams);")
        wait = body.index("SyncFunc<AscendC::HardEvent::MTE3_V>();", copy)
        self.assertLess(copy, wait)
        self.assertLess(wait, body.index("AccumulateSendCounts(dstRankTensor_"))

    def test_host_rejects_zero_bs(self):
        path = MC2 / "moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp"
        self.assertIn("(xDim0 <= 0) || (xDim0 > numMaxTokensPerRank)", path.read_text(encoding="utf-8"))

    def test_small_first_batch_then_full_batches(self):
        # Dispatch commits an exactly full batch immediately; Combine flushes on the next reserve or at the tail.
        for capacity, on_full in ((512, True), (256, False)):
            batch = AsyncBatch(capacity, on_full)
            for token in range(2000):
                batch.append((1, token), 4)
            batch.commit()
            self.assertEqual(batch.batch_sizes[0], FIRST_BATCH_WQEBB)
            self.assertIn(capacity, batch.batch_sizes[1:])
            self.assertTrue(all(size <= capacity for size in batch.batch_sizes))
            self.assertEqual(len(batch.submitted), 2000)

    def test_mixed_sge_peer_order_empty_slices_and_tail(self):
        for capacity, on_full in ((512, True), (256, False)):
            for peers in (1, 2, 3, 8, 33, 65, 1024):
                for owner in range(SENDERS):
                    batch, expected = AsyncBatch(capacity, on_full), []
                    for step in range(peers):
                        peer = (owner + step) % peers
                        if peer == 0:
                            continue  # Local copies and flags do not produce WQEs.
                        for token in range(peer % 7):
                            event = (peer, token)
                            batch.append(event, (1, 2, 4)[token % 3])
                            expected.append(event)
                        batch.append((peer, "flag"), 1)
                        expected.append((peer, "flag"))
                    batch.commit()
                    self.assertEqual(batch.submitted, expected)
                    self.assertEqual(batch.pending, [])
                    if batch.batch_sizes:
                        self.assertLessEqual(batch.batch_sizes[0], FIRST_BATCH_WQEBB)
                        self.assertTrue(all(size <= capacity for size in batch.batch_sizes))

    def test_submission_does_not_imply_incoming_completion(self):
        batch = AsyncBatch(512, True)
        batch.append((1, "payload"), 4)
        batch.append((1, "flag"), 1)
        batch.commit()
        incoming = [0] * (8 * SENDERS)
        self.assertEqual(len(batch.submitted), 2)  # Sender can return with an unfinished receive lane.
        self.assertFalse(all(incoming))
        incoming[:-1] = [1] * (len(incoming) - 1)
        self.assertFalse(all(incoming))
        incoming[-1] = 1
        self.assertTrue(all(incoming))  # Only now may the epilogue consume data and clear flags.

    def test_epilogue_flag_lifecycle_across_rounds(self):
        for peers in (1, 2, 8, 33, 65):
            flags = [0] * (peers * SENDERS)
            for iteration in range(70):
                order = list(range(len(flags)))
                if iteration % 2:
                    order.reverse()
                for index in order[:-1]:
                    flags[index] = 1
                self.assertFalse(all(flags))
                flags[order[-1]] = 1
                self.assertTrue(all(flags))
                flags[:] = [0] * len(flags)
                self.assertFalse(any(flags))


if __name__ == "__main__":
    unittest.main()
