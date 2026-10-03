"""GQA 分块打包（chunked packing）交叉验证：8:1（32:4）双路径位级一致。

官方 500 用例套件含 32:4（gqa_ratio=8）× 9 条；host tiling 按 MAX_PACK_ROWS=128
行上限将 8:1 拆为 2 个任务（每任务 4 Q 头 × 32 行）。本测试锁定分块语义：
kernel 宿主串行分支（DecodeTask 分块驱动）与孪生全流水执行器（头全局循环，
与分块无关）在 32:4 上输出与 LSE 逐位一致——chunks=1 的既有算术不变，
chunks=2 的解码/寻址/槽索引由本测试直接覆盖。
"""

import ctypes
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_kernel_twin as tk  # noqa: E402  （复用孪生 dylib 装载与 runner）
from sageattention2.dispatch import v_scale_max_for  # noqa: E402


def _run_kernel_host_gqa8(q, k, v, km, gran_code, causal):
    """kernel 宿主串行分支 @ gqa=8：m_per_core=32 → 2 chunk × 4 头 × 32 行。

    lse_correction 与孪生侧同源（tk._lse_correction 唯一定义，fp32 matmul q·km^T），
    两侧消费同值缓冲，LSE 位级可比。
    """
    b, hq, hkv, sq, sk, d = 1, 32, 4, q.shape[2], q.shape[3], q.shape[3]
    out_pad = torch.zeros_like(q)
    lse = torch.zeros(b * hq * sq, dtype=torch.float32)
    corr = tk._lse_correction(q, k, True).contiguous().view(-1)
    t = tk.SageAttention2TilingData()
    t.b, t.hq, t.hkv, t.sq, t.sk, t.d_og, t.d_pad = b, hq, hkv, sq, sk, d, 128
    t.is_causal, t.tensor_layout = causal, tk.LAYOUT_HND
    t.qk_quant_gran, t.pv_accum_dtype = gran_code, 2
    t.smooth_k, t.smooth_v = 1, 0
    t.input_dtype = tk._dtype_mode(q)
    t.sm_scale_log2e = (1.0 / math.sqrt(d)) * tk.LOG2E
    t.v_scale_max = v_scale_max_for("fp32+fp16")  # 分档唯一定义点 dispatch
    t.s_fp8_offset = tk.S_FP8_OFFSET
    t.m_per_core, t.block_n, t.k_step = 32, 64, 16
    t.core_num_used = 1
    t.q_blk_num = (sq + t.m_per_core - 1) // t.m_per_core
    t.kv_blk_num = (sk + t.block_n - 1) // t.block_n
    t.split_kv_num, t.is_merge = 1, 0
    t.l1_q_offset = 0
    t.l1_q_scale_offset = 16 * 1024
    t.l1_kv_offset = 256 * 1024
    t.l1_kv_scale_offset = 256 * 1024 + 16 * 1024
    t.ub_acc_offset = 0
    t.ub_softmax_offset = 64 * 1024
    t.ub_quant_offset = 96 * 1024
    t.workspace_lse_offset, t.workspace_vm_offset = 0, 0
    t.enable_die_affinity, t.die_num = 0, 1
    t.cores_per_die, t.tasks_per_die = 1, b * hkv * 2 * t.q_blk_num
    tk._LIB.sage_attention2(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(km.data_ptr()), ctypes.c_void_p(corr.data_ptr()),
        ctypes.c_void_p(out_pad.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
        None, ctypes.cast(ctypes.byref(t), ctypes.c_void_p),
    )
    return out_pad.view(b, hq, sq, d), lse.view(b, hq, sq)


class TestGqaChunkedPacking(unittest.TestCase):
    """32:4（8:1）分块路径 vs 孪生：out 与 LSE 逐位一致。"""

    def test_gqa8_chunk2_bitwise_parity(self):
        torch.manual_seed(7)
        b, hq, hkv, sq, sk, d = 1, 32, 4, 128, 128, 128
        q = torch.randn(b, hq, sq, d, dtype=torch.float16)
        k = torch.randn(b, hkv, sk, d, dtype=torch.float16)
        v = torch.randn_like(k)
        km = k.mean(dim=2).float().contiguous()
        for gran_name, gran_code in (("per_thread", tk.GRAN_PER_THREAD),
                                     ("per_warp", tk.GRAN_PER_WARP)):
            for causal in (0, 1):
                with self.subTest(gran=gran_name, causal=bool(causal)):
                    case = {"saved_shape": {"batch_size": b, "num_qo_heads": hq,
                                            "num_kv_heads": hkv, "qo_len": sq,
                                            "kv_len": sk, "head_dim": d},
                            "qk_quant_gran": gran_name, "pv_accum_dtype": "fp32+fp16",
                            "is_causal": bool(causal), "smooth_k": True,
                            "tensor_layout": "HND"}
                    out_t, lse_t = tk._run_twin(case, q, k, v)
                    out_k, lse_k = _run_kernel_host_gqa8(q, k, v, km, gran_code, causal)
                    self.assertTrue(torch.equal(out_t, out_k),
                                    f"{gran_name} causal={causal}: output bit mismatch")
                    self.assertTrue(torch.equal(lse_t, lse_k),
                                    f"{gran_name} causal={causal}: LSE bit mismatch")


if __name__ == "__main__":
    unittest.main()
