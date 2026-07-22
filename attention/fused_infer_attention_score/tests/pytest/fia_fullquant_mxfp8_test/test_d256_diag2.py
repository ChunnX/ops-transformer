import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import torch, torch_npu
import fia_fullquant_mxfp8_golden as g
g.D = 256; g.N_q = 8; g.N_kv = 8; g.B = 1
g.ACTUAL_SEQ_Q = [128]; g.ACTUAL_SEQ_KV = [128]
g.ENABLE_PA = False; g.ENABLE_ROPE = False
g.INPUT_LAYOUT = "TND"; g.SPARSE_MODE = 3; g.GRAPH_PATH = 0
torch.npu.set_device(0)
data = g.generate_data()
q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr, kr, bt = data
cpu_out, _ = g.cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, qr, kr)
npu_out, _ = g.npu_mxfp8_fa(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, bt, qr, kr)
cpu_out = cpu_out.cpu().float()
npu_out = npu_out.cpu().float()
cpu_tnd = cpu_out.squeeze(0).permute(1, 0, 2)
# Show actual values for token 0, first head
print("=== Token 0, Head 0 ===")
print("CPU D[0:8]:", cpu_tnd[0, 0, 0:8].tolist())
print("NPU D[0:8]:", npu_out[0, 0, 0:8].tolist())
print("CPU D[128:136]:", cpu_tnd[0, 0, 128:136].tolist())
print("NPU D[128:136]:", npu_out[0, 0, 128:136].tolist())
print()
print("=== Token 32, Head 0 ===")
print("CPU D[0:8]:", cpu_tnd[32, 0, 0:8].tolist())
print("NPU D[0:8]:", npu_out[32, 0, 0:8].tolist())
print("CPU D[128:136]:", cpu_tnd[32, 0, 128:136].tolist())
print("NPU D[128:136]:", npu_out[32, 0, 128:136].tolist())
# Check if NPU D[128:255] == NPU D[0:127]
print()
print("=== NPU D[128:255] == NPU D[0:127]? ===")
npu_d0 = npu_out[:, :, 0:128].flatten()
npu_d1 = npu_out[:, :, 128:256].flatten()
match = torch.isclose(npu_d0, npu_d1, atol=1e-6).sum().item()
print(f"NPU D0 == NPU D1: {match}/{npu_d0.numel()}")
print("NPU D[0:8] token0 h0:", npu_out[0, 0, 0:8].tolist())
print("NPU D[128:136] token0 h0:", npu_out[0, 0, 128:136].tolist())
