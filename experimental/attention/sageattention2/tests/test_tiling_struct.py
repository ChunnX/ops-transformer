"""
Python verification test for SageAttention2 TilingData struct.
Verifies struct size, alignment, field offsets, and serialization/deserialization.
"""

import ctypes
import struct
import unittest


class SageAttention2TilingData(ctypes.Structure):
    _fields_ = [
        # 1. shape 字段 (28 bytes)
        ("b", ctypes.c_uint32),
        ("hq", ctypes.c_uint32),
        ("hkv", ctypes.c_uint32),
        ("sq", ctypes.c_uint32),
        ("sk", ctypes.c_uint32),
        ("d_og", ctypes.c_uint32),
        ("d_pad", ctypes.c_uint32),
        # 2. 模式字段 (24 bytes)
        ("is_causal", ctypes.c_uint32),
        ("tensor_layout", ctypes.c_uint32),
        ("qk_quant_gran", ctypes.c_uint32),
        ("pv_accum_dtype", ctypes.c_uint32),
        ("smooth_k", ctypes.c_uint32),
        ("smooth_v", ctypes.c_uint32),
        # 3. 数值字段 (8 bytes)
        ("sm_scale_log2e", ctypes.c_float),
        ("v_scale_max", ctypes.c_float),
        # 4. 分块字段 (12 bytes)
        ("m_per_core", ctypes.c_uint32),
        ("block_n", ctypes.c_uint32),
        ("k_step", ctypes.c_uint32),
        # 5. 分核字段 (20 bytes)
        ("core_num_used", ctypes.c_uint32),
        ("q_blk_num", ctypes.c_uint32),
        ("kv_blk_num", ctypes.c_uint32),
        ("split_kv_num", ctypes.c_uint32),
        ("is_merge", ctypes.c_uint32),
        # 6. 布局与片上/全局偏移字段 (44 bytes)
        ("l1_q_offset", ctypes.c_uint32),
        ("l1_q_scale_offset", ctypes.c_uint32),
        ("l1_kv_offset", ctypes.c_uint32),
        ("l1_kv_scale_offset", ctypes.c_uint32),
        ("ub_acc_offset", ctypes.c_uint32),
        ("ub_softmax_offset", ctypes.c_uint32),
        ("ub_quant_offset", ctypes.c_uint32),
        ("workspace_lse_offset", ctypes.c_uint64),
        ("workspace_vm_offset", ctypes.c_uint64),
        # 7. 补齐保留字段 (24 bytes)
        ("reserved", ctypes.c_uint8 * 24),
    ]


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

        # 分块字段
        self.assertEqual(SageAttention2TilingData.m_per_core.offset, 60)
        self.assertEqual(SageAttention2TilingData.block_n.offset, 64)
        self.assertEqual(SageAttention2TilingData.k_step.offset, 68)

        # 分核字段
        self.assertEqual(SageAttention2TilingData.core_num_used.offset, 72)
        self.assertEqual(SageAttention2TilingData.q_blk_num.offset, 76)
        self.assertEqual(SageAttention2TilingData.kv_blk_num.offset, 80)
        self.assertEqual(SageAttention2TilingData.split_kv_num.offset, 84)
        self.assertEqual(SageAttention2TilingData.is_merge.offset, 88)

        # 布局与偏移字段
        self.assertEqual(SageAttention2TilingData.l1_q_offset.offset, 92)
        self.assertEqual(SageAttention2TilingData.l1_q_scale_offset.offset, 96)
        self.assertEqual(SageAttention2TilingData.l1_kv_offset.offset, 100)
        self.assertEqual(SageAttention2TilingData.l1_kv_scale_offset.offset, 104)
        self.assertEqual(SageAttention2TilingData.ub_acc_offset.offset, 108)
        self.assertEqual(SageAttention2TilingData.ub_softmax_offset.offset, 112)
        self.assertEqual(SageAttention2TilingData.ub_quant_offset.offset, 116)
        self.assertEqual(SageAttention2TilingData.workspace_lse_offset.offset, 120)
        self.assertEqual(SageAttention2TilingData.workspace_vm_offset.offset, 128)
        self.assertEqual(SageAttention2TilingData.reserved.offset, 136)

    def test_pack_unpack(self):
        # 验证 struct 打包与 ctypes 解析一致性
        fmt = "<7I 6I 2f 3I 5I 7I 2Q 24s"
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
        self.assertEqual(unpacked[30], 1048576)  # workspace_lse_offset


if __name__ == "__main__":
    unittest.main()
