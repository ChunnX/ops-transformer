# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU completion-accounting model; does not simulate Hcomm or NPU memory ordering."""

import unittest

SENDERS = 32
CQ_DEPTH = 64
SQ_DEPTH = 4096
BATCH_WQEBB = CQ_DEPTH // 2


def receive_lane(owner, peers):
    return [source * SENDERS + owner for source in range(peers)]


class BatchModel:
    def __init__(self, cq_counter=0, commit_on_full=False):
        self.pending = []
        self.pending_wqebb = 0
        self.submitted = []
        self.commit_on_full = commit_on_full
        self.cq_head = cq_counter
        self.cq_tail = cq_counter
        self.max_outstanding_cqe = 0
        self.max_batch_wqebb = 0
        self.commits = 0
        self.drains = 0

    def append(self, event, wqebb):
        if self.pending_wqebb + wqebb > BATCH_WQEBB:
            self.flush()
        self.pending.append(event)
        self.pending_wqebb += wqebb
        if self.commit_on_full and self.pending_wqebb >= BATCH_WQEBB:
            self.flush()

    def commit(self):
        if self.pending:
            self.submitted.extend(self.pending)
            self.cq_head += sum(event[2] for event in self.pending)
            self.max_outstanding_cqe = max(self.max_outstanding_cqe, self.cq_head - self.cq_tail)
            self.max_batch_wqebb = max(self.max_batch_wqebb, self.pending_wqebb)
            self.commits += 1
            self.pending.clear()
            self.pending_wqebb = 0

    def drain(self):
        if self.pending:
            raise RuntimeError("cannot drain an uncommitted tail")
        outstanding = self.cq_head - self.cq_tail
        self.cq_tail = self.cq_head
        self.drains += 1
        return outstanding

    def flush(self):
        if self.pending:
            self.commit()
            self.drain()


def send_round(peers, local, owner, payloads, wqebb, cq_counter=0, commit_on_full=False):
    batch = BatchModel(cq_counter, commit_on_full)
    for step in range(peers):
        peer = (owner + step) % peers
        if peer == local:
            continue  # Local completion is a GM write, not a WQE/CQE.
        for token in range(payloads[peer]):
            batch.append((peer, token, True), wqebb)
        batch.append((peer, "flag", True), 1)
    batch.flush()
    return batch


class SenderCompletionTest(unittest.TestCase):
    def test_every_remote_write_generates_cqe(self):
        for peers in (1, 2, 3, 8, 16, 32, 33, 64, 65, 128, 1024):
            for local in {0, peers // 2, peers - 1}:
                for owner in range(SENDERS):
                    batch = send_round(peers, local, owner, [1] * peers, 2)
                    expected = []
                    for step in range(peers):
                        peer = (owner + step) % peers
                        if peer != local:
                            expected.extend(((peer, 0, True), (peer, "flag", True)))
                    self.assertEqual(batch.submitted, expected)
                    self.assertEqual(batch.cq_head, 2 * (peers - 1))
                    self.assertEqual(batch.cq_tail, batch.cq_head)

    def test_each_commit_drains_within_sq_and_cq_capacity(self):
        for peers in (1, 2, 3, 8, 33, 65):
            for local in {0, peers - 1}:
                for owner in range(SENDERS):
                    for wqebb in (1, 2, 4):
                        for commit_on_full in (False, True):
                            for count in (0, 1, 8, 32, 257):
                                payloads = [count if peer == 0 else peer % 3 for peer in range(peers)]
                                batch = send_round(peers, local, owner, payloads, wqebb,
                                                   commit_on_full=commit_on_full)
                                flags = [event for event in batch.submitted if event[1] == "flag"]
                                expected = sum(payloads[p] + 1 for p in range(peers) if p != local)
                                self.assertEqual(len(flags), peers - 1)
                                self.assertEqual(batch.cq_head, expected)
                                self.assertEqual(batch.cq_head, batch.cq_tail)
                                self.assertEqual(batch.commits, batch.drains)
                                self.assertLessEqual(batch.max_outstanding_cqe, BATCH_WQEBB)
                                self.assertLessEqual(batch.max_batch_wqebb, BATCH_WQEBB)
                                self.assertLess(batch.max_outstanding_cqe, CQ_DEPTH)
                                self.assertLess(batch.max_batch_wqebb, SQ_DEPTH)

    def test_mixed_sge_batches_and_large_empty_peer_set(self):
        for commit_on_full in (False, True):
            batch = BatchModel(commit_on_full=commit_on_full)
            for token in range(8192):
                batch.append((1, token, True), (1, 2, 4)[token % 3])
            batch.flush()
            self.assertEqual(batch.cq_head, 8192)  # One CQE per WQE, not per SGE/WQEBB.
            self.assertEqual(batch.cq_head, batch.cq_tail)
            self.assertLessEqual(batch.max_batch_wqebb, BATCH_WQEBB)
            self.assertLessEqual(batch.max_outstanding_cqe, BATCH_WQEBB)
            batch = send_round(1024, 0, 31, [0] * 1024, 1, commit_on_full=commit_on_full)
            self.assertEqual(batch.cq_head, 1023)
            self.assertEqual(batch.cq_head, batch.cq_tail)
            self.assertEqual(batch.max_outstanding_cqe, BATCH_WQEBB)

    def test_count_phase_drains_before_recreating_handle(self):
        for peers in (2, 8, 33, 65, 1024):
            for local in {0, peers - 1}:
                for owner in range(SENDERS):
                    counter = CQ_DEPTH - 1
                    expected = counter
                    for peer in range(owner, peers, SENDERS):
                        if peer == local:
                            continue
                        batch = BatchModel(counter)
                        batch.append((peer, "count", True), 2)
                        batch.flush()
                        expected += 1
                        self.assertEqual(batch.cq_head, expected)
                        self.assertEqual(batch.cq_head, batch.cq_tail)
                        self.assertEqual(batch.max_outstanding_cqe, 1)
                        counter = batch.cq_tail
                    batch = send_round(peers, local, owner, [0] * peers, 1, counter)
                    self.assertEqual(batch.cq_tail, expected + peers - 1)

    def test_cq_wrap_and_empty_payload_reuse(self):
        for owner in range(SENDERS):
            counter = 0
            for iteration in range(3 * CQ_DEPTH + 1):
                payloads = [0] * 8 if iteration % 2 else [1] * 8
                batch = send_round(8, iteration % 8, owner, payloads, 2, counter)
                expected = counter + 7 * (1 + payloads[0])
                self.assertEqual(batch.cq_head, expected)
                self.assertEqual(batch.cq_tail, expected)
                counter = batch.cq_tail
                self.assertEqual(counter % CQ_DEPTH, expected % CQ_DEPTH)

    def test_receive_owners_cover_every_flag_once(self):
        for peers in (1, 2, 3, 8, 33, 65, 1024):
            for aivs in (33, 48, 56, 64, 80, 100):
                visits = [0] * (peers * SENDERS)
                for core in range(aivs):
                    if core < SENDERS:
                        for index in receive_lane(core, peers):
                            visits[index] += 1
                self.assertEqual(visits, [1] * len(visits))

    def test_local_drain_does_not_replace_receive_wait(self):
        for peers in (2, 8, 33):
            for owner in range(SENDERS):
                batch = send_round(peers, 0, owner, [0] * peers, 1)
                self.assertEqual(batch.cq_tail, peers - 1)
                flags = [1] * (peers * SENDERS)
                delayed = receive_lane(owner, peers)[-1]
                flags[delayed] = 0
                self.assertFalse(all(flags[i] == 1 for i in receive_lane(owner, peers)))
                flags[delayed] = 1
                self.assertTrue(all(flags[i] == 1 for i in receive_lane(owner, peers)))

    def test_sender_wait_keeps_flags_for_epilogue(self):
        for peers in (2, 8, 33):
            flags = [0] * (peers * SENDERS)
            for _ in range(10):
                for owner in range(SENDERS):
                    for index in receive_lane(owner, peers):
                        flags[index] = 1
                for owner in range(SENDERS):
                    self.assertTrue(all(flags[i] == 1 for i in receive_lane(owner, peers)))
                self.assertEqual(flags, [1] * len(flags))  # Epilogue can still observe all completions.
                flags[:] = [0] * len(flags)  # Exactly the existing epilogue clear, once per round.

    def test_pending_tail_must_be_committed(self):
        batch = BatchModel()
        batch.append((0, "flag", True), 1)
        with self.assertRaises(RuntimeError):
            batch.drain()
        batch.commit()
        self.assertEqual(batch.drain(), 1)


if __name__ == "__main__":
    unittest.main()
