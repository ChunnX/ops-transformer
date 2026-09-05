"""Adversarial CPU model of the persistent MTE barrier generation protocol.

The model uses unbounded logical rounds as an independent safety oracle and
32-bit wire values for peer observations. It does not model device coherence.
"""

import random
import unittest


MASK32 = (1 << 32) - 1


def reached(observed, target):
    """The device's signed modular comparison, without Python signed casts."""
    return ((observed - target) & MASK32) < (1 << 31)


class BarrierModel:
    def __init__(self, ranks, cores, rounds, initial):
        self.ranks, self.cores, self.rounds = ranks, cores, rounds
        self.initial = initial
        self.published = [0] * ranks  # Unbounded logical generations.
        self.wire = [initial] * ranks
        self.completed = [0] * ranks
        self.arrived = [set() for _ in range(ranks)]
        self.departed = [set() for _ in range(ranks)]
        self.next_peer = [0] * ranks
        self.phase = ["entry"] * ranks

    def actions(self):
        actions = []
        for rank in range(self.ranks):
            phase = self.phase[rank]
            if phase == "entry":
                actions.extend((rank, "arrive", core) for core in range(self.cores)
                               if core not in self.arrived[rank])
                if len(self.arrived[rank]) == self.cores:
                    actions.append((rank, "publish", None))
            elif phase == "poll":
                peer = self.next_peer[rank]
                if reached(self.wire[peer], self.wire[rank]):
                    actions.append((rank, "observe", peer))
            elif phase == "exit":
                actions.extend((rank, "depart", core) for core in range(self.cores)
                               if core not in self.departed[rank])
        return actions

    def step(self, action):
        rank, kind, index = action
        if kind == "arrive":
            self.arrived[rank].add(index)
        elif kind == "publish":
            assert len(self.arrived[rank]) == self.cores
            assert len(self.departed[rank]) == 0
            self.published[rank] += 1
            self.wire[rank] = (self.initial + self.published[rank]) & MASK32
            self.phase[rank] = "poll"
        elif kind == "observe":
            # Check the wire comparison against the independent logical oracle.
            assert self.published[index] >= self.published[rank]
            self.next_peer[rank] += 1
            if self.next_peer[rank] == self.ranks:
                assert min(self.published) >= self.published[rank]
                self.phase[rank] = "exit"
        elif kind == "depart":
            assert min(self.published) >= self.published[rank]
            self.departed[rank].add(index)
            if len(self.departed[rank]) == self.cores:
                self.completed[rank] += 1
                self.phase[rank] = "done" if self.completed[rank] == self.rounds else "entry"
                self.arrived[rank].clear()
                self.departed[rank].clear()
                self.next_peer[rank] = 0
        else:
            raise AssertionError(kind)
        assert max(self.published) - min(self.published) <= 1

    def run(self, seed):
        rng = random.Random(seed)
        # Each action advances a core, publishes once, or observes one peer.
        expected_steps = self.ranks * self.rounds * (2 * self.cores + 1 + self.ranks)
        for _ in range(expected_steps):
            actions = self.actions()
            assert actions, "All ranks stalled before completing the barrier sequence"
            self.step(rng.choice(actions))
        assert self.completed == [self.rounds] * self.ranks
        assert not self.actions()


class MteBarrierTests(unittest.TestCase):
    def test_rounds_under_rank_and_core_interleaving(self):
        for ranks, cores, rounds in ((1, 1, 12), (2, 4, 21), (8, 4, 15), (8, 72, 6)):
            for initial in (0, MASK32 - 1, MASK32):
                for seed in range(4):
                    with self.subTest(ranks=ranks, cores=cores, initial=initial, seed=seed):
                        BarrierModel(ranks, cores, rounds, initial).run(seed)

    def test_no_publish_until_every_local_aiv_arrives(self):
        model = BarrierModel(2, 4, 1, 0)
        for core in range(3):
            model.step((0, "arrive", core))
        self.assertNotIn((0, "publish", None), model.actions())
        model.step((0, "arrive", 3))
        self.assertIn((0, "publish", None), model.actions())

    def test_fast_peer_can_enter_next_round_before_slow_peer_observes(self):
        for initial in (0, MASK32 - 1, MASK32):
            with self.subTest(initial=initial):
                model = BarrierModel(2, 1, 2, initial)
                for rank in (0, 1):
                    model.step((rank, "arrive", 0))
                    model.step((rank, "publish", None))
                model.step((0, "observe", 0))
                model.step((0, "observe", 1))
                model.step((0, "depart", 0))
                model.step((0, "arrive", 0))
                model.step((0, "publish", None))
                self.assertNotEqual(model.wire[0], model.wire[1])
                self.assertIn((1, "observe", 0), model.actions())
                model.step((1, "observe", 0))
                model.step((1, "observe", 1))
                self.assertEqual(model.phase[1], "exit")

    def test_last_local_aiv_must_exit_before_next_publish(self):
        model = BarrierModel(1, 4, 2, 0)
        for core in range(4):
            model.step((0, "arrive", core))
        model.step((0, "publish", None))
        model.step((0, "observe", 0))
        for core in range(3):
            model.step((0, "depart", core))
        self.assertEqual(model.completed, [0])
        self.assertEqual(model.phase, ["exit"])
        model.step((0, "depart", 3))
        self.assertEqual(model.completed, [1])
        self.assertEqual(model.phase, ["entry"])

    def test_modular_comparison_at_wrap(self):
        for target in (0, 1, MASK32 - 1, MASK32):
            with self.subTest(target=target):
                self.assertTrue(reached(target, target))
                self.assertTrue(reached((target + 1) & MASK32, target))
                self.assertFalse(reached((target - 1) & MASK32, target))


if __name__ == "__main__":
    unittest.main()
