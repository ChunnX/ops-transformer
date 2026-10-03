"""SageAttention2TilingData 的 ctypes 镜像（唯一定义点）。

权威定义：sageattn_fp8/ascendc/tiling.h 的 SageAttention2TilingData
（160 字节、32 字节对齐、Host/Device 共享只读）。
两个消费方：
- tests/test_tiling_struct.py：40 个字段偏移断言（对镜像的逐字段锁）
- tests/test_kernel_twin.py：宿主串行分支的 tiling 填充
本模块 import 时自检 sizeof == struct.calcsize(FMT) == 160，作为第一道护栏。
任何字段改动必须先通过 tiling.h 的 static_assert 与 test_tiling_struct 的偏移断言。
"""

import ctypes
import struct

#: struct.pack 序列化格式（与 _fields_ 逐字段对应）
TILING_STRUCT_FMT = "<7I 6I 3f 3I 5I 7I 2Q 4I B3s"

#: 冻结结构体大小（字节）
EXPECTED_SIZE = 160


class SageAttention2TilingData(ctypes.Structure):
    _pack_ = 1
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
        # 3. 数值字段 (12 bytes)；s_fp8_offset 为布局冻结保留字段（kernel 不读）
        ("sm_scale_log2e", ctypes.c_float),
        ("v_scale_max", ctypes.c_float),
        ("s_fp8_offset", ctypes.c_float),
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
        # 7. Die 调度与硬件亲和性字段 (16 bytes)
        ("enable_die_affinity", ctypes.c_uint32),
        ("die_num", ctypes.c_uint32),
        ("cores_per_die", ctypes.c_uint32),
        ("tasks_per_die", ctypes.c_uint32),
        # 8. dtype 与补齐保留字段 (4 bytes)：input_dtype 输入位型分派（1=fp16, 2=bf16）
        ("input_dtype", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


# 第一道护栏：镜像自检（ctypes 布局 vs struct 格式 vs 冻结大小，三方一致）
assert ctypes.sizeof(SageAttention2TilingData) == struct.calcsize(TILING_STRUCT_FMT), (
    "tiling mirror: ctypes sizeof != struct.calcsize(TILING_STRUCT_FMT)"
)
assert ctypes.sizeof(SageAttention2TilingData) == EXPECTED_SIZE, (
    "tiling mirror: struct size drifted from the frozen 160-byte contract"
)
