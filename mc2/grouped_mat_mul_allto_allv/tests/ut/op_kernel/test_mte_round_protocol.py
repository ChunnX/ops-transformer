"""Explore ring-buffer rank barriers, including sequential peer polling.

Generation metadata is verification-only. Hardware stores just 0/1/2.
This checks mailbox lifetime, not device cache or pipeline visibility.
"""
from collections import deque
import unittest


def explore(ranks, rounds, buffers, exit_ack=True):
    # Positions: (invocation, phase, peer cursor). Slots: (value, generation).
    start = (((0, 0, 0),) * ranks, ((0, -1),) * (ranks * buffers),
             ((0, -1),) * (ranks * buffers))
    queue, seen = deque([start]), {start}
    deadlocks = unsafe = finished = 0
    while queue:
        pos, counts, dones = queue.popleft()
        successors = []
        for rank, (inv, phase, peer) in enumerate(pos):
            if inv == rounds:
                continue
            c, d = list(counts), list(dones)
            own, remote = (inv % buffers) * ranks + rank, (inv % buffers) * ranks + peer
            next_pos = (inv, phase + 1, 0)
            if phase == 0:
                c[own] = d[own] = (0, inv)
            elif phase == 1:
                c[own] = (1, inv)
            elif phase == 3:
                d[own] = (2, inv)  # Abstract all local payload pulls completed.
            elif phase == 5:
                c[own] = (0, inv)
            elif phase == 7:
                d[own] = (0, inv)
            else:
                expected = 1 if phase == 2 else 2 if phase == 4 else 0
                value, generation = (d if phase in (4, 8) else c)[remote]
                if value != expected:
                    continue
                if (value != 0 and generation != inv) or (value == 0 and generation < inv):
                    unsafe += 1
                if peer + 1 < ranks:
                    next_pos = (inv, phase, peer + 1)
                elif phase == (8 if exit_ack else 6):
                    next_pos = (inv + 1, 0, 0)
            p = list(pos)
            p[rank] = next_pos
            successors.append((tuple(p), tuple(c), tuple(d)))
        if not successors:
            if all(p[0] == rounds for p in pos):
                finished += 1
            else:
                deadlocks += 1
        for state in successors:
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return len(seen), deadlocks, unsafe, finished


class RoundProtocolTest(unittest.TestCase):
    def test_single_buffer_can_lose_exit_zero(self):
        _, deadlocks, unsafe, _ = explore(2, 3, 1, exit_ack=False)
        self.assertGreater(deadlocks + unsafe, 0)

    def test_exit_ack_also_protects_immediate_buffer_reuse(self):
        _, deadlocks, unsafe, finished = explore(3, 4, 1)
        self.assertEqual((deadlocks, unsafe), (0, 0))
        self.assertGreater(finished, 0)

    def test_two_buffers_survive_skew_and_reuse(self):
        for ranks, rounds in [(2, 5), (3, 3)]:
            with self.subTest(ranks=ranks, rounds=rounds):
                states, deadlocks, unsafe, finished = explore(ranks, rounds, 2)
                self.assertEqual((deadlocks, unsafe), (0, 0))
                self.assertGreater(finished, 0)
                print(f'ranks={ranks} rounds={rounds} states={states} deadlocks={deadlocks} unsafe={unsafe}')


if __name__ == '__main__':
    unittest.main()
