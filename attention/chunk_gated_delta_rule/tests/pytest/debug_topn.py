"""Find which elements have the largest NPU vs bf16_sim divergence."""
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

DEVICE_ID = 0
torch_npu.npu.set_device(DEVICE_ID)

B = 1; seqlen = 64; nk = 4; nv = 4; dk = 128; dv = 128; chunk_size = 64
actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
T = B * seqlen

torch.manual_seed(42)
q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device="npu:%s" % DEVICE_ID)
k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device="npu:%s" % DEVICE_ID)
v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device="npu:%s" % DEVICE_ID)
g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
beta = torch.rand((T, nv), dtype=torch.bfloat16, device="npu:%s" % DEVICE_ID)
q = torch.nn.functional.normalize(q, p=2, dim=-1)
k = torch.nn.functional.normalize(k, p=2, dim=-1)
scale = 1 / (dk ** 0.5)
initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device="npu:%s" % DEVICE_ID)

def bf16_mm(a, b, trans_b=False):
    a_bf = a.to(torch.bfloat16).to(torch.float32)
    b_bf = b.to(torch.bfloat16).to(torch.float32)
    r = a_bf @ b_bf.transpose(-1, -2) if trans_b else a_bf @ b_bf
    return r.to(torch.bfloat16).to(torch.float32)

def f32_mm(a, b, trans_b=False):
    return a @ b.transpose(-1, -2) if trans_b else a @ b

def compute(q_in, k_in, v_in, g_in, beta_in, scale_in, state_in, cs, mm_fn):
    initial_dtype = q_in.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q_in, k_in, v_in, beta_in, g_in)]
    bs, nh, seq_len, k_hd = k.shape; v_hd = v.shape[-1]
    pad = (cs - seq_len % cs) % cs
    q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad)); g = F.pad(g, (0, pad))
    total_len = seq_len + pad; q = q * scale_in
    v_beta = v * beta.unsqueeze(-1); k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, cs, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, cs)
    mask = torch.triu(torch.ones(cs, cs, dtype=torch.bool, device=q.device), diagonal=0)
    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    kkt = mm_fn(k_beta, k, trans_b=True)
    attn = -(kkt * decay_mask).masked_fill(mask, 0)
    for i in range(1, cs):
        row = attn[..., i, :i].clone(); sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(cs, dtype=attn.dtype, device=attn.device)
    v_inner = mm_fn(attn, v_beta)
    k_cd = mm_fn(attn, k_beta * g_cum.exp().unsqueeze(-1))
    q_prime = q * g_cum.exp().unsqueeze(-1)
    kg = k * (g_cum[:, :, :, -1, None] - g_cum).exp().unsqueeze(-1)
    qkt = mm_fn(q, k, trans_b=True)
    last_state = state_in.to(v_inner)
    core_out = torch.zeros_like(v_inner)
    mask2 = torch.triu(torch.ones(cs, cs, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(total_len // cs):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v_inner[:, :, i]
        attn_inter = mm_fn(q_prime[:, :, i], last_state, trans_b=True)
        v_prime = mm_fn(k_cd[:, :, i], last_state, trans_b=True)
        v_new = v_i - v_prime
        attn_intra = (qkt[:, :, i] * decay_mask[:, :, i]).masked_fill_(mask2, 0)
        core_out[:, :, i] = attn_inter + mm_fn(attn_intra, v_new)
        last_state = last_state * g_cum[:, :, i, -1, None, None].exp() + mm_fn(v_new.transpose(-1, -2), kg[:, :, i])
    core_out = core_out.reshape(core_out.shape[0], core_out.shape[1], -1, core_out.shape[-1])
    core_out = core_out[:, :, :seq_len].transpose(1, 2).contiguous().to(initial_dtype)
    return core_out[0], last_state[0]

v_f32 = v.to(torch.float32)
q_t = q.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
k_t = k.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
v_t = v_f32.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
g_t = g.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
beta_t = beta.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
state_t = initial_state.transpose(-1, -2).clone().to(v_f32.device).to(v_f32.dtype)
if nv // nk > 1:
    q_t = q_t.repeat_interleave(nv // nk, dim=2)
    k_t = k_t.repeat_interleave(nv // nk, dim=2)
cur_q = q_t[:, 0:seqlen]; cur_k = k_t[:, 0:seqlen]; cur_v = v_t[:, 0:seqlen]
cur_g = g_t[:, 0:seqlen]; cur_beta = beta_t[:, 0:seqlen]; cur_state = state_t[0].unsqueeze(0)

o_f32, s_f32 = compute(cur_q, cur_k, cur_v, cur_g, cur_beta, scale, cur_state, chunk_size, f32_mm)
o_bf16, s_bf16 = compute(cur_q, cur_k, cur_v, cur_g, cur_beta, scale, cur_state, chunk_size, bf16_mm)

from chunk_gated_delta_rule_main import cgdr_npu
o_npu, s_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

o_f32_f = o_f32.to(torch.float32).cpu().numpy()
o_bf16_f = o_bf16.to(torch.float32).cpu().numpy()
o_npu_f = o_npu.to(torch.float32).cpu().numpy()

print("Output shape:", o_f32_f.shape, o_npu_f.shape)

# Find top-10 divergent elements (bf16_sim vs NPU)
diff_bn = np.abs(o_bf16_f - o_npu_f)
top_idx = np.argsort(diff_bn.flatten())[-20:][::-1]
print("\n===== Top-20 divergent elements (bf16_sim vs NPU) =====")
print(f"{'idx':>8} {'seq':>4} {'head':>4} {'dv':>4} {'f32':>12} {'bf16_sim':>12} {'npu':>12} {'diff_bf16_npu':>14} {'rel_diff':>10}")
for idx in top_idx:
    seq, head, d = np.unravel_index(idx, o_f32_f.shape)
    f32_v = o_f32_f[seq, head, d]
    bf16_v = o_bf16_f[seq, head, d]
    npu_v = o_npu_f[seq, head, d]
    diff = abs(bf16_v - npu_v)
    rel = diff / (abs(f32_v) + 1e-10)
    print(f"{idx:8d} {seq:4d} {head:4d} {d:4d} {f32_v:12.6f} {bf16_v:12.6f} {npu_v:12.6f} {diff:14.6f} {rel:10.4f}")

# Check if divergent elements are concentrated in specific heads
print("\n===== Divergence by head (bf16_sim vs NPU, elements with abs_diff > 0.01) =====")
for h in range(nv):
    mask_h = diff_bn[:, h, :] > 0.01
    cnt = mask_h.sum()
    max_d = diff_bn[:, h, :].max()
    print(f"  Head {h}: {cnt} elements > 0.01, max_diff={max_d:.6f}")

# Check if divergent elements are concentrated in specific sequence positions
print("\n===== Divergence by seq position (bf16_sim vs NPU, elements with abs_diff > 0.01) =====")
for s in range(seqlen):
    mask_s = diff_bn[s, :, :] > 0.01
    cnt = mask_s.sum()
    if cnt > 0:
        max_d = diff_bn[s, :, :].max()
        print(f"  Seq {s}: {cnt} elements > 0.01, max_diff={max_d:.6f}")

# g_cumsum analysis
g_cpu = g[0:64, :].cpu().numpy()
g_cum = np.cumsum(g_cpu, axis=0)
print(f"\n===== g_cumsum analysis =====")
print(f"g_cumsum[-1, :] = {g_cum[-1, :]}")
print(f"exp(g_cumsum[-1, :]) = {np.exp(g_cum[-1, :])}")
print(f"g_cumsum[0, :] = {g_cum[0, :]}")
print(f"exp(g_cumsum[-1] - g_cumsum[0]) = {np.exp(g_cum[-1, :] - g_cum[0, :])}")

# Check if the issue is the g_cumsum[-1] being very large negative
# This causes exp(g_cumsum) to underflow, and kg to be ~0
# In bfloat16, these small values might be flushed to 0
print(f"\n===== bfloat16 representation of kg =====")
kg_vals = np.exp(g_cum[-1, :] - g_cum[0, :])
print(f"kg scale factors (exp(g[-1]-g[0])): {kg_vals}")
print(f"These are {'< 1e-38 (bfloat16 underflow)' if kg_vals.min() < 1e-38 else 'representable in bfloat16'}")

# Test: what if g is smaller (less negative)?
print("\n===== Test with smaller g values =====")
g_small = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -0.1  # max -0.1 instead of -1.0
o_f32_s, s_f32_s = compute(cur_q, cur_k, cur_v, g_small.unsqueeze(0)[:, 0:seqlen].to(v_f32.device).to(v_f32.dtype),
                            cur_beta, scale, cur_state, chunk_size, f32_mm)
o_bf16_s, s_bf16_s = compute(cur_q, cur_k, cur_v, g_small.unsqueeze(0)[:, 0:seqlen].to(v_f32.device).to(v_f32.dtype),
                              cur_beta, scale, cur_state, chunk_size, bf16_mm)
o_npu_s, s_npu_s = cgdr_npu(q, k, v, g_small, beta, scale, initial_state, actual_seq_lengths)

diff_f32_bf16_s = np.abs(o_f32_s.to(torch.float32).cpu().numpy() - o_bf16_s.to(torch.float32).cpu().numpy())
diff_f32_npu_s = np.abs(o_f32_s.to(torch.float32).cpu().numpy() - o_npu_s.to(torch.float32).cpu().numpy())
diff_bf16_npu_s = np.abs(o_bf16_s.to(torch.float32).cpu().numpy() - o_npu_s.to(torch.float32).cpu().numpy())

print(f"With g*0.1 (smaller g):")
print(f"  f32 vs bf16_sim:  max_abs={diff_f32_bf16_s.max():.8f}")
print(f"  f32 vs NPU:       max_abs={diff_f32_npu_s.max():.8f}")
print(f"  bf16_sim vs NPU:  max_abs={diff_bf16_npu_s.max():.8f}")

rtol_bf, atol_bf = 0.0078125, 0.0001
fail_s = np.sum(diff_f32_npu_s > atol_bf + rtol_bf * np.abs(o_f32_s.to(torch.float32).cpu().numpy()))
total_s = diff_f32_npu_s.size
print(f"  f32_vs_npu bf16_fail: {fail_s}/{total_s} ({100*fail_s/total_s:.2f}%)")
