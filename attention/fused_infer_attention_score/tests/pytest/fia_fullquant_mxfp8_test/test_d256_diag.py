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
print(f"cpu shape: {cpu_out.shape}, npu shape: {npu_out.shape}")
# cpu_out is [1, 8, 128, 256] (B, heads, tokens, D)
# npu_out is [128, 8, 256] (tokens, heads, D) - TND layout
# Reshape cpu to match npu: permute to [tokens, heads, D]
cpu_tnd = cpu_out.squeeze(0).permute(1, 0, 2)  # [8,128,256] -> [128,8,256]
print(f"cpu_tnd shape: {cpu_tnd.shape}")
# Overall
cf = cpu_tnd.flatten(); nf = npu_out.flatten()
ic = torch.isclose(cf, nf, rtol=0.005, atol=0.000025)
print(f"Overall: pass={ic.sum().item()}/{ic.numel()} = {ic.sum().item()/ic.numel()*100:.2f}%")
# By D half
for ds, de, label in [(0, 128, "D[0:127]"), (128, 256, "D[128:255]")]:
    c = cpu_tnd[:, :, ds:de].flatten()
    n = npu_out[:, :, ds:de].flatten()
    ic2 = torch.isclose(c, n, rtol=0.005, atol=0.000025)
    print(f"{label}: pass={ic2.sum().item()}/{ic2.numel()} = {ic2.sum().item()/ic2.numel()*100:.2f}%")
# By token group x D half
for ts, te in [(0, 32), (32, 64), (64, 96), (96, 128)]:
    for ds, de, label in [(0, 128, "D0"), (128, 256, "D1")]:
        c = cpu_tnd[ts:te, :, ds:de].flatten()
        n = npu_out[ts:te, :, ds:de].flatten()
        ic3 = torch.isclose(c, n, rtol=0.005, atol=0.000025)
        p = ic3.sum().item(); t = ic3.numel()
        print(f"  T[{ts}:{te}] {label}: {p}/{t} = {p/t*100:.1f}%")
