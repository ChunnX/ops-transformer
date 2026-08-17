import unittest

from stem_indexer_metadata_cpu import (
    AIC_CORE_NUM,
    HEAD_METADATA_STRIDE,
    build_stem_indexer_schedule,
    generate_stem_indexer_metadata_values,
)


class TestStemIndexerMetadataCpu(unittest.TestCase):
    def test_single_block_matches_a6_metadata_layout(self):
        values, schedule = generate_stem_indexer_metadata_values(
            [128],
            [128],
            q_heads=1,
            kv_heads=1,
            causal=False,
            dim_qkflat=2048,
            return_schedule=True,
        )

        self.assertEqual(len(values), 4096)
        self.assertEqual(values[0], 1)
        self.assertEqual(schedule.section_num, 1)
        self.assertEqual(schedule.sections[0].used_core_num, 1)
        self.assertEqual(values[HEAD_METADATA_STRIDE : HEAD_METADATA_STRIDE + 6], [0, 0, 0, 1, 0, 0])

    def test_noncausal_two_m_blocks_are_not_split_on_s2(self):
        schedule = build_stem_indexer_schedule(
            [128 * 192],
            [128 * 256],
            q_heads=1,
            kv_heads=1,
            causal=False,
            dim_qkflat=2048,
        )

        section = schedule.sections[0]
        self.assertEqual(section.used_core_num, 2)
        self.assertEqual(section.cores[0].start, (0, 0, 0))
        self.assertEqual(section.cores[0].end, (0, 1, 0))
        self.assertEqual(section.cores[1].start, (0, 1, 0))
        # AssignByBatch normalizes the completed final M block to the next BN2.
        self.assertEqual(section.cores[1].end, (1, 0, 0))

    def test_l2_partition_creates_one_section_per_large_bn2(self):
        schedule = build_stem_indexer_schedule(
            [128 * 192, 128 * 192],
            [128 * 256, 128 * 256],
            q_heads=2,
            kv_heads=2,
            causal=False,
            dim_qkflat=65536,
        )

        self.assertEqual(schedule.section_num, 4)
        self.assertEqual([section.bn2_end for section in schedule.sections], [1, 2, 3, 4])

    def test_a6_performance_shape_uses_96_by_256_and_64_cores(self):
        schedule = build_stem_indexer_schedule(
            [65536],
            [262144],
            q_heads=64,
            kv_heads=8,
            causal=True,
            dim_qkflat=2048,
            window_size=4,
        )

        self.assertEqual(schedule.m_base_size, 96)
        self.assertEqual(schedule.s2_base_size, 256)
        # One BN2 consumes 20 MiB of the 96 MiB L2 budget, so eight heads form two sections.
        self.assertEqual(schedule.section_num, 2)
        self.assertEqual([section.bn2_end for section in schedule.sections], [4, 8])
        self.assertEqual([section.used_core_num for section in schedule.sections], [AIC_CORE_NUM, AIC_CORE_NUM])

    def test_rejects_non_divisible_head_group(self):
        with self.assertRaisesRegex(ValueError, "q_heads must be divisible by kv_heads"):
            build_stem_indexer_schedule([128], [128], q_heads=3, kv_heads=2)


if __name__ == "__main__":
    unittest.main()
