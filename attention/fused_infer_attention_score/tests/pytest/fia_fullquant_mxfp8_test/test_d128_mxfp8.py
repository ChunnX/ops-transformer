#!/usr/bin/python3
"""D=128 MxFP8 baseline test"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import torch, torch_npu
import fia_fullquant_mxfp8_golden as g

g.D = 128; g.N_q = 8; g.N_kv = 8; g.B = 1
g.ACTUAL_SEQ_Q = [128]; g.ACTUAL_SEQ_KV = [128]
g.ENABLE_PA = False; g.ENABLE_ROPE = False
g.INPUT_LAYOUT = "TND"; g.SPARSE_MODE = 3; g.GRAPH_PATH = 0

torch.npu.set_device(0)
data = g.generate_data()
q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr, kr, bt = data
cpu_out, _ = g.cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale,
                                 g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, qr, kr)
npu_out, _ = g.npu_mxfp8_fa(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale,
                              g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, bt, qr, kr)
cpu_tnd = g.convert_q_bnsd_to_layout(cpu_out, g.ACTUAL_SEQ_Q, "TND")
import numpy as _np; _np.save("/tmp/npu_out_d128.npy", _np.asarray(npu_out.cpu())); _np.save("/tmp/cpu_out_d128.npy", _np.asarray(cpu_tnd.cpu())); g.result_compare_method.check_result(cpu_tnd, npu_out)
