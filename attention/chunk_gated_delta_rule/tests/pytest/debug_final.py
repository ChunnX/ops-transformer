"""Quick focused analysis: f32 vs bf16_sim vs NPU final comparison."""
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

# Prepare inputs
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

o_f32_f = o_f32.to(torch.float32).cpu().numpy().flatten()
o_bf16_f = o_bf16.to(torch.float32).cpu().numpy().flatten()
o_npu_f = o_npu.to(torch.float32).cpu().numpy().flatten()
s_f32_f = s_f32.to(torch.float32).cpu().numpy().flatten()
s_bf16_f = s_bf16.to(torch.float32).cpu().numpy().flatten()
s_npu_f = s_npu.to(torch.float32).cpu().numpy().flatten()

print("===== Final Output Comparison =====")
print(f"f32 vs bf16_sim:  max_abs={np.abs(o_f32_f - o_bf16_f).max():.8f}  mean_abs={np.abs(o_f32_f - o_bf16_f).mean():.8f}")
print(f"f32 vs NPU:       max_abs={np.abs(o_f32_f - o_npu_f).max():.8f}  mean_abs={np.abs(o_f32_f - o_npu_f).mean():.8f}")
print(f"bf16_sim vs NPU:  max_abs={np.abs(o_bf16_f - o_npu_f).max():.8f}  mean_abs={np.abs(o_bf16_f - o_npu_f).mean():.8f}")

print("\n===== Final State Comparison =====")
print(f"f32 vs bf16_sim:  max_abs={np.abs(s_f32_f - s_bf16_f).max():.8f}  mean_abs={np.abs(s_f32_f - s_bf16_f).mean():.8f}")
print(f"f32 vs NPU:       max_abs={np.abs(s_f32_f - s_npu_f).max():.8f}  mean_abs={np.abs(s_f32_f - s_npu_f).mean():.8f}")
print(f"bf16_sim vs NPU:  max_abs={np.abs(s_bf16_f - s_npu_f).max():.8f}  mean_abs={np.abs(s_bf16_f - s_npu_f).mean():.8f}")

# isclose analysis
rtol_bf, atol_bf = 0.0078125, 0.0001
rtol_strict, atol_strict = 0.005, 0.000025
for label, diff, ref in [("f32_vs_npu", np.abs(o_f32_f - o_npu_f), o_f32_f),
                          ("bf16_sim_vs_npu", np.abs(o_bf16_f - o_npu_f), o_bf16_f)]:
    fail_bf = np.sum(diff > atol_bf + rtol_bf * np.abs(ref))
    fail_strict = np.sum(diff > atol_strict + rtol_strict * np.abs(ref))
    total = len(diff)
    print(f"\n  Output {label}: bf16_fail={fail_bf}/{total} ({100*fail_bf/total:.2f}%)  strict_fail={fail_strict}/{total} ({100*fail_strict/total:.2f}%)")

for label, diff, ref in [("f32_vs_npu", np.abs(s_f32_f - s_npu_f), s_f32_f),
                          ("bf16_sim_vs_npu", np.abs(s_bf16_f - s_npu_f), s_bf16_f)]:
    fail_bf = np.sum(diff > atol_bf + rtol_bf * np.abs(ref))
    fail_strict = np.sum(diff > atol_strict + rtol_strict * np.abs(ref))
    total = len(diff)
    print(f"  State  {label}: bf16_fail={fail_bf}/{total} ({100*fail_bf/total:.2f}%)  strict_fail={fail_strict}/{total} ({100*fail_strict/total:.2f}%)")

# Check data_type_str bug impact
print("\n===== check_result threshold bug =====")
print(f"str(torch.bfloat16) = '{str(torch.bfloat16)}' != 'bfloat16'")
print(f"  -> Uses strict threshold (rtol=0.005, atol=0.000025) instead of bfloat16 (rtol=0.0078125, atol=0.0001)")
print(f"  -> This makes the test FAIL more easily than intended")

# g_cumsum analysis
print("\n===== g_cumsum analysis =====")
g_cpu = g[0:64, 0].cpu().numpy()
g_cum = np.cumsum(g_cpu)
print(f"g_cumsum range: [{g_cum.min():.4f}, {g_cum.max():.4f}]")
print(f"exp(g_cumsum) range: [{np.exp(g_cum.min()):.2e}, {np.exp(g_cum.max()):.2e}]")
print(f"kg = k * exp(g[-1] - g[:]), g[-1]={g_cum[-1]:.4f}")
print(f"exp(g[-1] - g[0]) = {np.exp(g_cum[-1] - g_cum[0]):.2e} (essentially 0 in bfloat16)")
