"""
Python verification test for SageAttention2 TilingData struct.
Verifies struct size, alignment, field offsets, and serialization/deserialization.
"""

import ctypes
import struct
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ 内共享模块

# 镜像唯一定义点：tests/tiling_mirror.py（逐字段偏移由下方断言锁定）
from tiling_mirror import SageAttention2TilingData  # noqa: E402


class TestTilingDataStruct(unittest.TestCase):
    def test_size_and_alignment(self):
        # 结构体总大小必须严格为 160 字节且为 32 字节整数倍
        total_size = ctypes.sizeof(SageAttention2TilingData)
        self.assertEqual(total_size, 160)
        self.assertEqual(total_size % 32, 0)

    def test_field_offsets(self):
        # Shape 字段
        self.assertEqual(SageAttention2TilingData.b.offset, 0)
        self.assertEqual(SageAttention2TilingData.hq.offset, 4)
        self.assertEqual(SageAttention2TilingData.hkv.offset, 8)
        self.assertEqual(SageAttention2TilingData.sq.offset, 12)
        self.assertEqual(SageAttention2TilingData.sk.offset, 16)
        self.assertEqual(SageAttention2TilingData.d_og.offset, 20)
        self.assertEqual(SageAttention2TilingData.d_pad.offset, 24)

        # 模式字段
        self.assertEqual(SageAttention2TilingData.is_causal.offset, 28)
        self.assertEqual(SageAttention2TilingData.tensor_layout.offset, 32)
        self.assertEqual(SageAttention2TilingData.qk_quant_gran.offset, 36)
        self.assertEqual(SageAttention2TilingData.pv_accum_dtype.offset, 40)
        self.assertEqual(SageAttention2TilingData.smooth_k.offset, 44)
        self.assertEqual(SageAttention2TilingData.smooth_v.offset, 48)

        # 数值字段
        self.assertEqual(SageAttention2TilingData.sm_scale_log2e.offset, 52)
        self.assertEqual(SageAttention2TilingData.v_scale_max.offset, 56)
        self.assertEqual(SageAttention2TilingData.s_fp8_offset.offset, 60)

        # 分块字段
        self.assertEqual(SageAttention2TilingData.m_per_core.offset, 64)
        self.assertEqual(SageAttention2TilingData.block_n.offset, 68)
        self.assertEqual(SageAttention2TilingData.k_step.offset, 72)

        # 分核字段
        self.assertEqual(SageAttention2TilingData.core_num_used.offset, 76)
        self.assertEqual(SageAttention2TilingData.q_blk_num.offset, 80)
        self.assertEqual(SageAttention2TilingData.kv_blk_num.offset, 84)
        self.assertEqual(SageAttention2TilingData.split_kv_num.offset, 88)
        self.assertEqual(SageAttention2TilingData.is_merge.offset, 92)

        # 布局与偏移字段
        self.assertEqual(SageAttention2TilingData.l1_q_offset.offset, 96)
        self.assertEqual(SageAttention2TilingData.l1_q_scale_offset.offset, 100)
        self.assertEqual(SageAttention2TilingData.l1_kv_offset.offset, 104)
        self.assertEqual(SageAttention2TilingData.l1_kv_scale_offset.offset, 108)
        self.assertEqual(SageAttention2TilingData.ub_acc_offset.offset, 112)
        self.assertEqual(SageAttention2TilingData.ub_softmax_offset.offset, 116)
        self.assertEqual(SageAttention2TilingData.ub_quant_offset.offset, 120)
        self.assertEqual(SageAttention2TilingData.workspace_lse_offset.offset, 124)
        self.assertEqual(SageAttention2TilingData.workspace_vm_offset.offset, 132)
        self.assertEqual(SageAttention2TilingData.enable_die_affinity.offset, 140)
        self.assertEqual(SageAttention2TilingData.die_num.offset, 144)
        self.assertEqual(SageAttention2TilingData.cores_per_die.offset, 148)
        self.assertEqual(SageAttention2TilingData.tasks_per_die.offset, 152)
        self.assertEqual(SageAttention2TilingData.input_dtype.offset, 156)
        self.assertEqual(SageAttention2TilingData.reserved.offset, 157)

    def test_pack_unpack(self):
        # 验证 struct 打包与 ctypes 解析一致性
        fmt = "<7I 6I 3f 3I 5I 7I 2Q 4I B3s"
        self.assertEqual(struct.calcsize(fmt), 160)

        tiling = SageAttention2TilingData()
        tiling.b = 2
        tiling.hq = 32
        tiling.hkv = 8
        tiling.sq = 1024
        tiling.sk = 1024
        tiling.d_og = 128
        tiling.d_pad = 128
        tiling.is_causal = 0
        tiling.tensor_layout = 1
        tiling.qk_quant_gran = 3
        tiling.pv_accum_dtype = 2
        tiling.smooth_k = 1
        tiling.smooth_v = 0
        tiling.sm_scale_log2e = 0.1275
        tiling.v_scale_max = 2.25
        tiling.s_fp8_offset = 8.807355
        tiling.m_per_core = 64
        tiling.block_n = 64
        tiling.k_step = 16
        tiling.core_num_used = 16
        tiling.q_blk_num = 16
        tiling.kv_blk_num = 16
        tiling.split_kv_num = 1
        tiling.is_merge = 0
        tiling.l1_q_offset = 0
        tiling.l1_q_scale_offset = 16384
        tiling.l1_kv_offset = 262144
        tiling.l1_kv_scale_offset = 278528
        tiling.ub_acc_offset = 0
        tiling.ub_softmax_offset = 65536
        tiling.ub_quant_offset = 0
        tiling.workspace_lse_offset = 1048576
        tiling.workspace_vm_offset = 0
        tiling.input_dtype = 1

        raw_bytes = bytes(tiling)
        self.assertEqual(len(raw_bytes), 160)

        unpacked = struct.unpack(fmt, raw_bytes)
        self.assertEqual(unpacked[0], 2)  # b
        self.assertEqual(unpacked[1], 32)  # hq
        self.assertEqual(unpacked[2], 8)  # hkv
        self.assertEqual(unpacked[8], 1)  # tensor_layout
        self.assertEqual(unpacked[9], 3)  # qk_quant_gran
        self.assertEqual(unpacked[10], 2)  # pv_accum_dtype
        self.assertAlmostEqual(unpacked[13], 0.1275, places=5)
        self.assertAlmostEqual(unpacked[14], 2.25, places=5)
        self.assertAlmostEqual(unpacked[15], 8.807355, places=5)
        self.assertEqual(unpacked[31], 1048576)  # workspace_lse_offset
        self.assertEqual(unpacked[37], 1)  # input_dtype (1=fp16)
        self.assertEqual(unpacked[38], b"\x00\x00\x00")  # reserved


if __name__ == "__main__":
    unittest.main()
