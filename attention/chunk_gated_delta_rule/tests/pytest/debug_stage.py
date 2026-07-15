"""
Simulate NPU bfloat16 matmul precision in torch reference.
At each matmul boundary, quantize inputs/outputs to bfloat16 to match NPU behavior.
Compare float32 reference vs bfloat16-simulated reference to find error source.
"""
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

DEVICE_ID = 0
torch_npu.npu.set_device(DEVICE_ID)

B = 1
seqlen = 64
nk = 4
nv = 4
dk = 128
dv = 128
chunk_size = 64
data_type = torch.bfloat16

actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
T = B * seqlen

torch.manual_seed(42)
q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
q = torch.nn.functional.normalize(q, p=2, dim=-1)
k = torch.nn.functional.normalize(k, p=2, dim=-1)
scale = 1 / (dk ** 0.5)
initial_state = torch.rand((B, nv, dv, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)


def bf16_matmul(a, b, trans_b=False):
    """Simulate NPU bfloat16 matmul: inputs and output are bfloat16."""
    a_bf = a.to(torch.bfloat16).to(torch.float32)
    b_bf = b.to(torch.bfloat16).to(torch.float32)
    if trans_b:
        result = a_bf @ b_bf.transpose(-1, -2)
    else:
        result = a_bf @ b_bf
    return result.to(torch.bfloat16).to(torch.float32)


def f32_matmul(a, b, trans_b=False):
    """Pure float32 matmul (torch reference)."""
    if trans_b:
        return a @ b.transpose(-1, -2)
    return a @ b


def compute_gdr(q_in, k_in, v_in, g_in, beta_in, scale_in, state_in, chunk_size_in, matmul_fn, label=""):
    """Compute chunk_gated_delta_rule with specified matmul precision."""
    initial_dtype = q_in.dtype
    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q_in, k_in, v_in, beta_in, g_in)
    ]

    bs, nh, seq_len, k_hd = k.shape
    v_hd = v.shape[-1]
    pad = (chunk_size_in - seq_len % chunk_size_in) % chunk_size_in
    q = F.pad(q, (0, 0, 0, pad))
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    q = q * scale_in

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size_in, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size_in)
    mask = torch.triu(torch.ones(chunk_size_in, chunk_size_in, dtype=torch.bool, device=q.device), diagonal=0)

    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()

    # Stage1: kkt = k_beta @ k.T  (bfloat16 matmul on NPU)
    kkt = matmul_fn(k_beta, k, trans_b=True)
    attn = -(kkt * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size_in):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size_in, dtype=attn.dtype, device=attn.device)

    # v_inner = attn @ v_beta  (bfloat16 matmul)
    v_inner = matmul_fn(attn, v_beta)

    # k_cumdecay = attn @ (k_beta * g.exp())  (bfloat16 matmul)
    k_cumdecay = matmul_fn(attn, k_beta * g_cum.exp().unsqueeze(-1))

    # q_prime = q * scale * g.exp()  (vector op, float32 on NPU)
    q_prime = q * g_cum.exp().unsqueeze(-1)

    # kg = k * (g[-1] - g).exp()  (vector op)
    kg = k * (g_cum[:, :, :, -1, None] - g_cum).exp().unsqueeze(-1)

    # qkt = q @ k.T  (bfloat16 matmul, without scale)
    qkt = matmul_fn(q, k, trans_b=True)

    last_state = state_in.to(v_inner)
    core_out = torch.zeros_like(v_inner)
    mask2 = torch.triu(torch.ones(chunk_size_in, chunk_size_in, dtype=torch.bool, device=q.device), diagonal=1)

    intermediates = {}

    for i in range(0, total_len // chunk_size_in):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v_inner[:, :, i]

        # Stage2: attn_inter = q_prime @ state.T  (bfloat16 matmul)
        attn_inter = matmul_fn(q_prime[:, :, i], last_state, trans_b=True)

        # Stage2: v_prime = k_cumdecay @ state.T  (bfloat16 matmul)
        v_prime = matmul_fn(k_cumdecay[:, :, i], last_state, trans_b=True)
        v_new = v_i - v_prime

        # Stage3: attn_intra = (qkt * decay_mask).masked_fill  (vector + bfloat16)
        attn_intra = (qkt[:, :, i] * decay_mask[:, :, i]).masked_fill_(mask2, 0)

        # Stage3: out += attn_intra @ v_new  (bfloat16 matmul)
        attn_vnew = matmul_fn(attn_intra, v_new)
        core_out[:, :, i] = attn_inter + attn_vnew

        if i == 0:
            intermediates['kkt'] = kkt[0].cpu().numpy()
            intermediates['attn_inv'] = attn[0].cpu().numpy()
            intermediates['v_inner'] = v_inner[0].cpu().numpy()
            intermediates['k_cumdecay'] = k_cumdecay[0].cpu().numpy()
            intermediates['q_prime'] = q_prime[0].cpu().numpy()
            intermediates['kg'] = kg[0].cpu().numpy()
            intermediates['qkt'] = qkt[0].cpu().numpy()
            intermediates['attn_inter'] = attn_inter[0].cpu().numpy()
            intermediates['v_prime'] = v_prime[0].cpu().numpy()
            intermediates['v_new'] = v_new[0].cpu().numpy()
            intermediates['attn_intra'] = attn_intra[0].cpu().numpy()
            intermediates['attn_vnew'] = attn_vnew[0].cpu().numpy()
            intermediates['core_out_chunk0'] = core_out[0, :, 0].cpu().numpy()
            intermediates['state_before'] = last_state[0].cpu().numpy()

        # state update: state = state * exp(g[-1]) + kg.T @ v_new  (bfloat16 matmul)
        # NPU: state += v_new.T @ kg  (after state *= exp(g[-1]))
        state_scaled = last_state * g_cum[:, :, i, -1, None, None].exp()
        state_update = matmul_fn(v_new.transpose(-1, -2), kg[:, :, i])
        last_state = state_scaled + state_update

        if i == 0:
            intermediates['state_scaled'] = state_scaled[0].cpu().numpy()
            intermediates['state_update'] = state_update[0].cpu().numpy()
            intermediates['state_after'] = last_state[0].cpu().numpy()

    core_out = core_out.reshape(core_out.shape[0], core_out.shape[1], -1, core_out.shape[-1])
    core_out = core_out[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    intermediates['final_out'] = core_out[0].cpu().numpy()
    intermediates['final_state'] = last_state[0].cpu().numpy()
    return core_out, last_state, intermediates


# Prepare inputs
cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
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

cur_q = q_t[:, 0:seqlen]
cur_k = k_t[:, 0:seqlen]
cur_v = v_t[:, 0:seqlen]
cur_g = g_t[:, 0:seqlen]
cur_beta = beta_t[:, 0:seqlen]
cur_state = state_t[0].unsqueeze(0)

# Run float32 reference
print("===== Float32 Reference =====")
o_f32, s_f32, inter_f32 = compute_gdr(cur_q, cur_k, cur_v, cur_g, cur_beta, scale, cur_state, chunk_size, f32_matmul, "f32")

# Run bfloat16-simulated reference
print("===== Bfloat16 Simulated =====")
o_bf16, s_bf16, inter_bf16 = compute_gdr(cur_q, cur_k, cur_v, cur_g, cur_beta, scale, cur_state, chunk_size, bf16_matmul, "bf16")

# Run NPU
print("===== NPU =====")
from chunk_gated_delta_rule_main import cgdr_npu
o_npu, s_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

# Compare stage by stage
print("\n===== Stage-by-stage comparison (f32 vs bf16_sim) =====")
keys = ['kkt', 'attn_inv', 'v_inner', 'k_cumdecay', 'q_prime', 'kg', 'qkt',
        'attn_inter', 'v_prime', 'v_new', 'attn_intra', 'attn_vnew',
        'core_out_chunk0', 'state_before', 'state_scaled', 'state_update', 'state_after',
        'final_out', 'final_state']

for key in keys:
    f32_val = inter_f32[key]
    bf16_val = inter_bf16[key]
    diff = np.abs(f32_val - bf16_val)
    rel_diff = diff / (np.abs(f32_val) + 1e-10)
    print(f"  {key:25s}: shape={f32_val.shape}  max_abs={diff.max():.8f}  max_rel={rel_diff.max():.8f}  mean_abs={diff.mean():.8f}")

# Compare final outputs: f32 vs bf16_sim vs NPU
print("\n===== Final Output Comparison =====")
o_f32_flat = o_f32[0].to(torch.float32).cpu().numpy().flatten()
o_bf16_flat = o_bf16[0].to(torch.float32).cpu().numpy().flatten()
o_npu_flat = o_npu[0].to(torch.float32).cpu().numpy().flatten()

diff_f32_bf16 = np.abs(o_f32_flat - o_bf16_flat)
diff_f32_npu = np.abs(o_f32_flat - o_npu_flat)
diff_bf16_npu = np.abs(o_bf16_flat - o_npu_flat)

print(f"  f32 vs bf16_sim:  max_abs={diff_f32_bf16.max():.8f}  mean_abs={diff_f32_bf16.mean():.8f}")
print(f"  f32 vs NPU:       max_abs={diff_f32_npu.max():.8f}  mean_abs={diff_f32_npu.mean():.8f}")
print(f"  bf16_sim vs NPU:  max_abs={diff_bf16_npu.max():.8f}  mean_abs={diff_bf16_npu.mean():.8f}")

# State comparison
print("\n===== Final State Comparison =====")
s_f32_flat = s_f32[0].to(torch.float32).cpu().numpy().flatten()
s_bf16_flat = s_bf16[0].to(torch.float32).cpu().numpy().flatten()
s_npu_flat = s_npu[0].to(torch.float32).cpu().numpy().flatten()

diff_s_f32_bf16 = np.abs(s_f32_flat - s_bf16_flat)
diff_s_f32_npu = np.abs(s_f32_flat - s_npu_flat)
diff_s_bf16_npu = np.abs(s_bf16_flat - s_npu_flat)

print(f"  f32 vs bf16_sim:  max_abs={diff_s_f32_bf16.max():.8f}  mean_abs={diff_s_f32_bf16.mean():.8f}")
print(f"  f32 vs NPU:       max_abs={diff_s_f32_npu.max():.8f}  mean_abs={diff_s_f32_npu.mean():.8f}")
print(f"  bf16_sim vs NPU:  max_abs={diff_s_bf16_npu.max():.8f}  mean_abs={diff_s_bf16_npu.mean():.8f}")

# Check isclose with bfloat16 thresholds
rtol, atol = 0.0078125, 0.0001
for label, diff, ref in [("f32_vs_npu", diff_f32_npu, o_f32_flat),
                          ("bf16_sim_vs_npu", diff_bf16_npu, o_bf16_flat)]:
    fail = np.sum(diff > atol + rtol * np.abs(ref))
    total = len(diff)
    print(f"\n  {label}: {fail}/{total} ({100*fail/total:.2f}%) fail bfloat16 isclose")

# Find which stage contributes most to f32 vs bf16_sim divergence
print("\n===== Key finding: which matmul contributes most error? =====")
for key in ['kkt', 'attn_inv', 'v_inner', 'k_cumdecay', 'qkt', 'attn_inter', 'v_prime', 'attn_vnew', 'state_update']:
    f32_val = inter_f32[key].flatten()
    bf16_val = inter_bf16[key].flatten()
    rel_diff = np.abs(f32_val - bf16_val) / (np.abs(f32_val) + 1e-10)
    print(f"  {key:25s}: max_rel_diff={rel_diff.max():.6f}  mean_rel_diff={np.mean(rel_diff[rel_diff < 1]):.6f}")
