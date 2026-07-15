import os
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import struct

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
state_data_type = torch.bfloat16
has_g = True
is_continue = True

actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
T = B * seqlen

torch.manual_seed(42)
q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
if has_g:
    g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
else:
    g = None
beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
q = torch.nn.functional.normalize(q, p=2, dim=-1)
k = torch.nn.functional.normalize(k, p=2, dim=-1)
scale = 1 / (dk ** 0.5)
initial_state = torch.rand((B, nv, dv, dk), dtype=state_data_type, device="npu:%s" % DEVICE_ID)

print(f"Inputs: B={B} seqlen={seqlen} nk={nk} nv={nv} dk={dk} dv={dv} chunk_size={chunk_size}")

# Save inputs for NPU-side comparison
torch.save({'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g': g.cpu() if g is not None else None,
            'beta': beta.cpu(), 'initial_state': initial_state.cpu(), 'scale': scale,
            'actual_seq_lengths': actual_seq_lengths.cpu()}, 'debug_inputs.pt')

# ======================== Torch reference with intermediate dump ========================
def torch_ref_with_dump(q_in, k_in, v_in, g_in, beta_in, scale_in, initial_state_in, chunk_size_in=64):
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    v_f32 = v_in.to(torch.float32)
    if g_in is None:
        g_in = torch.zeros((v_f32.shape[0], v_f32.shape[1])).to(v_f32.device).to(v_f32.dtype)

    q_t = q_in.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
    k_t = k_in.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
    v_t = v_f32.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
    g_t = g_in.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
    beta_t = beta_in.unsqueeze(0).to(v_f32.device).to(v_f32.dtype)
    state_t = initial_state_in.transpose(-1, -2).clone().to(v_f32.device).to(v_f32.dtype)

    num_heads = q_t.shape[-2]
    num_value_heads = v_t.shape[-2]
    if num_value_heads // num_heads > 1:
        q_t = q_t.repeat_interleave(num_value_heads // num_heads, dim=2)
        k_t = k_t.repeat_interleave(num_value_heads // num_heads, dim=2)

    batch_size = state_t.shape[0]
    core_attn_out_list = []
    last_recurrent_state = torch.empty_like(state_t)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q_t[:, start:end, ...]
        cur_k = k_t[:, start:end, ...]
        cur_v = v_t[:, start:end, ...]
        cur_g = g_t[:, start:end, ...]
        cur_beta = beta_t[:, start:end, ...]
        cur_state = state_t[b_idx].unsqueeze(0)

        result = compute_chunk_gdr(cur_q, cur_k, cur_v, cur_g, cur_beta, scale_in, cur_state, chunk_size_in)
        core_attn_out_list.append(result['out'])
        last_recurrent_state[b_idx] = result['final_state']

    tar_dtype = core_attn_out_list[0].dtype
    tar_device = core_attn_out_list[0].device
    tar_shape = list(core_attn_out_list[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)
    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_out[:, start:end, ...] = core_attn_out_list[b_idx]

    final_out = final_out[0].to(torch.float32).npu()
    final_state = last_recurrent_state.transpose(-1, -2).to(torch.float32).npu()
    return final_out, final_state


def compute_chunk_gdr(query, key, value, g, beta, scale, initial_state, chunk_size=64):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v_inner = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    q_prime = query * g_cum.exp().unsqueeze(-1)
    kg = key * (g_cum[:, :, :, -1, None] - g_cum).exp().unsqueeze(-1)

    # qkt = query @ key.T (scaled)
    qkt = query @ key.transpose(-1, -2)

    intermediates = {
        'g_cumsum': g_cum[0].cpu().numpy(),
        'decay_mask': decay_mask[0].cpu().numpy(),
        'attn_inv': attn[0].cpu().numpy(),
        'v_inner': v_inner[0].cpu().numpy(),
        'k_cumdecay': k_cumdecay[0].cpu().numpy(),
        'q_prime': q_prime[0].cpu().numpy(),
        'kg': kg[0].cpu().numpy(),
        'qkt': qkt[0].cpu().numpy(),
    }

    last_recurrent_state = initial_state.to(v_inner)
    core_attn_out = torch.zeros_like(v_inner)
    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], v_inner[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask2, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_cum[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new

        last_recurrent_state = (
            last_recurrent_state * g_cum[:, :, i, -1, None, None].exp()
            + (k_i * (g_cum[:, :, i, -1, None] - g_cum[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    intermediates['out_after_stage2'] = core_attn_out[0].cpu().numpy()
    intermediates['final_state'] = last_recurrent_state[0].cpu().numpy()

    result = {'out': core_attn_out, 'final_state': last_recurrent_state, 'intermediates': intermediates}
    return result


# ======================== Run torch reference ========================
print("\n========== Running torch reference ==========")
o_torch, state_torch = torch_ref_with_dump(q, k, v, g, beta, scale, initial_state, chunk_size)

# Re-run to get intermediates
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

cur_q = q_t[:, 0:seqlen, ...]
cur_k = k_t[:, 0:seqlen, ...]
cur_v = v_t[:, 0:seqlen, ...]
cur_g = g_t[:, 0:seqlen, ...]
cur_beta = beta_t[:, 0:seqlen, ...]
cur_state = state_t[0].unsqueeze(0)
result = compute_chunk_gdr(cur_q, cur_k, cur_v, cur_g, cur_beta, scale, cur_state, chunk_size)
torch_inter = result['intermediates']
np.savez('torch_intermediates.npz', **torch_inter)

print(f"\n[Torch] g_cumsum shape={torch_inter['g_cumsum'].shape}")
print(f"[Torch] g_cumsum[0,:8] = {torch_inter['g_cumsum'].flatten()[:8]}")
print(f"[Torch] attn_inv[0,0,:4,:4] = \n{torch_inter['attn_inv'][0,:4,:4]}")
print(f"[Torch] v_inner[0,0,:4,:4] = \n{torch_inter['v_inner'][0,:4,:4]}")
print(f"[Torch] k_cumdecay[0,0,:4,:4] = \n{torch_inter['k_cumdecay'][0,0,:4,:4]}")
print(f"[Torch] q_prime[0,0,:4,:4] = \n{torch_inter['q_prime'][0,0,:4,:4]}")
print(f"[Torch] kg[0,0,:4,:4] = \n{torch_inter['kg'][0,0,:4,:4]}")
print(f"[Torch] qkt[0,0,:4,:4] = \n{torch_inter['qkt'][0,0,:4,:4]}")
print(f"[Torch] out_after_stage2[0,:4,:4] = \n{torch_inter['out_after_stage2'][0,:4,:4]}")
print(f"[Torch] final_state[0,:4,:4] = \n{torch_inter['final_state'][0,:4,:4]}")

# ======================== Run NPU ========================
print("\n========== Running NPU ==========")
from chunk_gated_delta_rule_main import cgdr_npu
o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
print(f"[NPU] output[0,:4,:4] = \n{o_npu[0,:4,:4].cpu().to(torch.float32).numpy()}")
print(f"[NPU] state[0,:4,:4] = \n{state_npu[0,:4,:4].cpu().to(torch.float32).numpy()}")

# ======================== Compare final outputs ========================
print("\n========== Final Output Comparison ==========")
o_diff = (o_torch.to(torch.float32) - o_npu.to(torch.float32)).abs()
state_diff = (state_torch.to(torch.float32) - state_npu.to(torch.float32)).abs()
print(f"Output: max_abs={o_diff.max().item():.6f} mean_abs={o_diff.mean().item():.6f}")
print(f"State:  max_abs={state_diff.max().item():.6f} mean_abs={state_diff.mean().item():.6f}")

# Check rtol threshold
o_rtol = 0.0078125
o_atol = 0.0001
o_fail = ((o_diff > o_atol + o_rtol * o_npu.to(torch.float32).abs()).sum().item())
o_total = o_diff.numel()
print(f"Output: {o_fail}/{o_total} ({100*o_fail/o_total:.2f}%) elements fail bfloat16 isclose")

s_fail = ((state_diff > o_atol + o_rtol * state_npu.to(torch.float32).abs()).sum().item())
s_total = state_diff.numel()
print(f"State:  {s_fail}/{s_total} ({100*s_fail/s_total:.2f}%) elements fail bfloat16 isclose")

print("\n>>> Now run: bash build.sh --pkg --ops=chunk_gated_delta_rule --soc=ascend950 -j32")
print(">>> Then install and re-run this script to get NPU dump data")
print(">>> NPU dump files will be in core_id_0/ directory")
