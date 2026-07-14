import torch
import torch_npu
import torch.nn.functional as F

def stage1_opt(
    query,  # (S, Nk, Dk)
    key,    # (S, Nk, Dk)
    value,  # (S, Nv, Dv)
    g,      # (S, Nv)
    beta,   # (S, Nv)
    scale,
    C       # chunk size
):
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device=query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C
    n_chunks = padded_seq_len // C
    pad_size = padded_seq_len - S

    # Pre-pad all inputs along seq dim (dim 0)
    def pad_seq(t):
        if pad_size > 0:
            pad_pattern = [0, 0] * (t.dim() - 1) + [0, pad_size]
            return F.pad(t, pad_pattern)
        return t

    q_pad = pad_seq(query)       # (Sp, Nv, Dk)
    k_pad = pad_seq(key)         # (Sp, Nv, Dk)
    v_pad = pad_seq(value)       # (Sp, Nv, Dv)
    g_pad = pad_seq(g)           # (Sp, Nv)
    b_pad = pad_seq(beta)        # (Sp, Nv)

    # Reshape to (Nv*nc, C, D) — batch over all heads & chunks, replacing double Python loop
    q_all = q_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    k_all = k_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    v_all = v_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dv)
    g_all = g_pad.permute(1, 0).reshape(Nv * n_chunks, C)
    b_all = b_pad.permute(1, 0).reshape(Nv * n_chunks, C)

    # Batched computation
    k_f = k_all.float()
    b_f = b_all.float().unsqueeze(-1)

    kkt = (k_all @ k_all.transpose(-1, -2)).float() * b_f    # (Nv*nc, C, C)
    qkt = q_all @ k_all.transpose(-1, -2)                     # (Nv*nc, C, C)

    g_cum = g_all.cumsum(dim=-1)                               # (Nv*nc, C)
    g_cum_exp = g_cum.exp()                                    # (Nv*nc, C)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
    diff = g_cum[:, :, None] - g_cum[:, None, :]              # (Nv*nc, C, C)
    attn = (diff * lower).exp() * lower                        # (Nv*nc, C, C)

    eye = torch.eye(C, dtype=torch.float32, device=device)
    attn_1 = torch.linalg.solve_triangular(eye + kkt * attn, eye, upper=False)
    attn_1 = attn_1.to(torch.bfloat16)                         # (Nv*nc, C, C)

    kg = k_f * (g_cum[:, -1, None] - g_cum).exp()[..., None]  # (Nv*nc, C, Dk)
    k_cumdecay = b_f * g_cum_exp[:, :, None] * (-1) * k_f     # (Nv*nc, C, Dk)

    v_beta = (v_all.float() * b_f).to(torch.bfloat16)          # (Nv*nc, C, Dv)
    q_prime = (q_all.float() * scale * g_cum_exp[:, :, None]).to(torch.bfloat16)  # (Nv*nc, C, Dk)

    v_inner = attn_1 @ v_beta                                  # (Nv*nc, C, Dv)
    k_cumdecay = attn_1 @ k_cumdecay.to(torch.bfloat16)        # (Nv*nc, C, Dk)
    kg = kg.to(torch.bfloat16)                                 # (Nv*nc, C, Dk)

    # Reshape back to (Nv, Sp, ...)
    g_cum = g_cum.reshape(Nv, padded_seq_len)
    k_cumdecay = k_cumdecay.reshape(Nv, padded_seq_len, Dk)
    v_inner = v_inner.reshape(Nv, padded_seq_len, Dv)
    q_prime = q_prime.reshape(Nv, padded_seq_len, Dk)
    kg = kg.reshape(Nv, padded_seq_len, Dk)
    qkt = qkt.reshape(Nv, padded_seq_len, C)

    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt

def stage2_opt(
    q_prime,    # (Nv, Sp, Dk)
    v_inner,    # (Nv, Sp, Dv)
    g_cum,  # (Nv, Sp)
    k_cumdecay, # (Nv, Sp, Dk)
    state,      # (Nv, Dv, Dk)
    kg,         # (Nv, Sp, Dk)
    C,           # chunk size
):
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    n_chunks = Sp // C
    device = q_prime.device

    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)

    # Batch over Nv, loop only over chunks (sequential state dependency between chunks)
    cur_state = state  # (Nv, Dv, Dk)
    for idx in range(n_chunks):
        s = idx * C
        e = s + C

        qg = q_prime[:, s:e, :]       # (Nv, C, Dk)
        vi = v_inner[:, s:e, :]       # (Nv, C, Dv)
        gc = g_cum[:, s:e]            # (Nv, C)
        kcd = k_cumdecay[:, s:e, :]   # (Nv, C, Dk)
        kgc = kg[:, s:e, :]           # (Nv, C, Dk)

        # Batched stage2 computation over all heads
        state_f = cur_state.float()              # (Nv, Dv, Dk)
        state_fT = state_f.transpose(-1, -2)    # (Nv, Dk, Dv)

        attn_inter_chunk = (qg.float() @ state_fT).to(torch.bfloat16)    # (Nv, C, Dv)
        v_prime = (kcd.float() @ state_fT).to(torch.bfloat16)            # (Nv, C, Dv)
        v_new_chunk = vi + v_prime                                         # (Nv, C, Dv)

        state_out = v_new_chunk.transpose(-1, -2) @ kgc                   # (Nv, Dv, Dk)
        decay = gc.exp()[:, -1][:, None, None]                            # (Nv, 1, 1)
        state_old = (state_f * decay).to(torch.bfloat16)                  # (Nv, Dv, Dk)
        cur_state = state_old + state_out                                  # (Nv, Dv, Dk)

        attn_inter[:, s:e, :] = attn_inter_chunk
        v_new[:, s:e, :] = v_new_chunk

    final_state = cur_state.to(torch.bfloat16)
    return final_state, attn_inter, v_new

def stage3_opt(
    qkt,         # (Nv, Sp, C)
    scale,       # float
    g_cum,   # (Nv, Sp)
    attn_inter,  # (Nv, Sp, Dv)
    v_new,       # (Nv, Sp, Dv)
    C,            # chunk size
):
    Nv, Sp, Dv = attn_inter.shape
    assert Sp % C == 0
    n_chunks = Sp // C
    device = qkt.device

    # Reshape all inputs to (Nv*nc, C, ...) — batch over all heads & chunks
    qkt_all = qkt.reshape(Nv * n_chunks, C, C)
    g_all = g_cum.reshape(Nv * n_chunks, C)
    ai_all = attn_inter.reshape(Nv * n_chunks, C, Dv)
    vn_all = v_new.reshape(Nv * n_chunks, C, Dv)

    # Batched computation (replaces double loop over Nv and chunks)
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    diff = g_all[:, :, None] - g_all[:, None, :]              # (Nv*nc, C, C)
    decay = (diff * lower).exp() * lower.float()               # (Nv*nc, C, C)
    masked_qkt = qkt_all.float() * scale * decay               # (Nv*nc, C, C)
    attn_inner = masked_qkt.to(torch.bfloat16) @ vn_all        # (Nv*nc, C, Dv)
    core_attn_out = (ai_all + attn_inner).to(torch.bfloat16)   # (Nv*nc, C, Dv)

    # Reshape back: (Nv*nc, C, Dv) -> (Nv, Sp, Dv) -> (Sp, Nv, Dv)
    attn_out = core_attn_out.reshape(Nv, Sp, Dv).permute(1, 0, 2)

    return attn_out

def chunk_gdn_benchmark_opt(
    query,              # (T, Nk, Dk)
    key,                # (T, Nk, Dk)
    value,              # (T, Nv, Dv)
    beta,               # (T, Nv)
    scale,              # float
    initial_state,      # (B, Nv, Dv, Dk)
    actual_seq_lengths, # (B,)
    g = None            # (T, Nv)
):
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device=query.device

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=query.dtype, device=device)
    attn_out = (attn_out).to(torch.bfloat16)
    final_state = torch.empty_like(initial_state).to(torch.bfloat16)

    start = 0
    C = 64
    for bid in range(B):
        cur_state = initial_state[bid].clone()
        S = actual_seq_lengths[bid]
        end = start + S

        g_cum, k_cum_decay, v_inner, q_prime, kg, qkt = stage1_opt(
            query[start:end], key[start:end], value[start:end], g[start:end], beta[start:end], scale, C)

        cur_state, attn_inter, v_new = stage2_opt(
            q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C)
        final_state[bid] = cur_state.to(torch.bfloat16)
        attn_out_paddend = stage3_opt(
            qkt, scale, g_cum, attn_inter, v_new, C)
        
        attn_out[start:end, ...] = attn_out_paddend[:S]
        start = end

    return attn_out, final_state



# 用例
def A5_Redline_cases():
    # name, B, T, Nv, Nk, Dv, Dk, has_g, scale
    test_cases = [
        ("ARC-001", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-002", 1, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-003", 1, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-004", 2, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-005", 4, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-006", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-007", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-008", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-009", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-010", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-011", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-012", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-013", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-014", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-015", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-016", 32, 256000, 8, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-017", 1, 6000, 8, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-018", 80, 400000, 16, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-019", 200, 51200, 8, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-020", 1, 10240, 8, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-021", 1, 10240, 16, 4, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-022", 1, 10240, 32, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-023", 1, 10240, 64, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-024", 1, 32768, 8, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-025", 1, 32768, 16, 4, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-026", 1, 32768, 32, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-027", 1, 32768, 64, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-028", 4, 40960, 4, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-029", 4, 40960, 8, 4, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-030", 4, 40960, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-031", 1, 32768, 4, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-032", 1, 32768, 8, 4, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-033", 1, 32768, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-034", 1, 65535, 4, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-035", 1, 65535, 8, 4, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-036", 1, 65535, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-037", 1, 32768, 48, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ARC-038", 1, 32768, 16, 16, 128, 128, True, 1.0, torch.bfloat16),
    ]
    return test_cases

def A5_Redline_cases_fp32():
    # name, B, T, Nv, Nk, Dv, Dk, has_g, scale
    test_cases = [
        ("ARC-001-FP32", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-002-FP32", 1, 4096, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-003-FP32", 1, 8192, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-004-FP32", 2, 4096, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-005-FP32", 4, 8192, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-006-FP32", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-007-FP32", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-008-FP32", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-009-FP32", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-010-FP32", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ARC-011-FP32", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-012-FP32", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-013-FP32", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-014-FP32", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-015-FP32", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-016-FP32", 32, 256000, 8, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-017-FP32", 1, 6000, 8, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-018-FP32", 80, 400000, 16, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-019-FP32", 200, 51200, 8, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-020-FP32", 1, 10240, 8, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-021-FP32", 1, 10240, 16, 4, 128, 128, True, 1.0, torch.float32),
        ("ARC-022-FP32", 1, 10240, 32, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-023-FP32", 1, 10240, 64, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-024-FP32", 1, 32768, 8, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-025-FP32", 1, 32768, 16, 4, 128, 128, True, 1.0, torch.float32),
        ("ARC-026-FP32", 1, 32768, 32, 8, 128, 128, True, 1.0, torch.float32),
        ("ARC-027-FP32", 1, 32768, 64, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-028-FP32", 4, 40960, 4, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-029-FP32", 4, 40960, 8, 4, 128, 128, True, 1.0, torch.float32),
        ("ARC-030-FP32", 4, 40960, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-031-FP32", 1, 32768, 4, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-032-FP32", 1, 32768, 8, 4, 128, 128, True, 1.0, torch.float32),
        ("ARC-033-FP32", 1, 32768, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-034-FP32", 1, 65535, 4, 2, 128, 128, True, 1.0, torch.float32),
        ("ARC-035-FP32", 1, 65535, 8, 4, 128, 128, True, 1.0, torch.float32),
        ("ARC-036-FP32", 1, 65535, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-037-FP32", 1, 32768, 48, 16, 128, 128, True, 1.0, torch.float32),
        ("ARC-038-FP32", 1, 32768, 16, 16, 128, 128, True, 1.0, torch.float32),
    ]
    return test_cases

def A5_STC_cases():
    # name, B, T, Nv, Nk, Dv, Dk, has_g, scale
    test_cases = [
        ("ASC-001", 2, 2*4*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-002", 4, 4*8*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-003", 1, 32*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-004", 2, 2*32*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-005", 3, 3*32*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-006", 4, 4*32*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-007", 1, 64*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-008", 2, 2*64*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-009", 3, 3*64*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-010", 4, 4*64*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-011", 1, 128*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-012", 1, 256*1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-013", 1, 2*1024, 16, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-014", 1, 2*1024, 2, 2, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-015", 1, 2*1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-016", 1, 2*1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-017", 1, 4*1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-018", 1, 8*1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-019", 1, 128, 32, 32, 128, 64, True, 1.0, torch.bfloat16),
        ("ASC-020", 4, 4*512, 8, 8, 64, 128, True, 1.0, torch.bfloat16),
        ("ASC-021", 2, 2*1024, 32, 4, 64, 64, True, 1.0, torch.bfloat16),
        ("ASC-022", 1, 4096, 64, 1, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-023", 3, 3*10000, 64, 16, 50, 50, True, 1.0, torch.bfloat16),
        ("ASC-024", 3, 3*10000, 32, 16, 8, 5, True, 1.0, torch.bfloat16),
        ("ASC-025", 2, 2*300, 8, 8, 64, 64, False, 1.0, torch.bfloat16),
        ("ASC-026", 4, 4*512, 8, 8, 1, 1, False, 1.0, torch.bfloat16),
        ("ASC-027", 4, 4*1024, 8, 8, 64, 64, True, 1.0, torch.bfloat16),
        ("ASC-028", 4, 4*512, 8, 4, 64, 64, False, 1.0, torch.bfloat16),
        ("ASC-029", 4, 4*512, 16, 4, 64, 64, False, 1.0, torch.bfloat16),
        ("ASC-030", 8, 8*4096, 64, 64, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-031", 4, 4*128, 16, 8, 31, 31, True, 1.0, torch.bfloat16),
        ("ASC-032", 8, 8*128, 2, 2, 16, 16, True, 1.0, torch.bfloat16),
        ("ASC-033", 1, 128, 3, 3, 128, 128, True, 1.0, torch.bfloat16),
        ("ASC-034", 4, 4*2000, 2, 2, 16, 16, True, 1.0, torch.bfloat16),
    ]
    return test_cases

def A5_STC_cases_fp32():
    # name, B, T, Nv, Nk, Dv, Dk, has_g, scale
    test_cases = [
        ("ASC-001-FP32", 2, 2*4*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-002-FP32", 4, 4*8*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-003-FP32", 1, 32*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-004-FP32", 2, 2*32*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-005-FP32", 3, 3*32*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-006-FP32", 4, 4*32*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-007-FP32", 1, 64*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-008-FP32", 2, 2*64*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-009-FP32", 3, 3*64*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-010-FP32", 4, 4*64*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-011-FP32", 1, 128*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-012-FP32", 1, 256*1024, 32, 32, 128, 128, True, 1.0, torch.float32),
        ("ASC-013-FP32", 1, 2*1024, 16, 16, 128, 128, True, 1.0, torch.float32),
        ("ASC-014-FP32", 1, 2*1024, 2, 2, 128, 128, True, 1.0, torch.float32),
        ("ASC-015-FP32", 1, 2*1024, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ASC-016-FP32", 1, 2*1024, 16, 8, 128, 128, True, 1.0, torch.float32),
        ("ASC-017-FP32", 1, 4*1024, 32, 16, 128, 128, True, 1.0, torch.float32),
        ("ASC-018-FP32", 1, 8*1024, 16, 8, 128, 128, True, 1.0, torch.float32),
        ("ASC-019-FP32", 1, 128, 32, 32, 128, 64, True, 1.0, torch.float32),
        ("ASC-020-FP32", 4, 4*512, 8, 8, 64, 128, True, 1.0, torch.float32),
        ("ASC-021-FP32", 2, 2*1024, 32, 4, 64, 64, True, 1.0, torch.float32),
        ("ASC-022-FP32", 1, 4096, 64, 1, 128, 128, True, 1.0, torch.float32),
        ("ASC-023-FP32", 3, 3*10000, 64, 16, 50, 50, True, 1.0, torch.float32),
        ("ASC-024-FP32", 3, 3*10000, 32, 16, 8, 5, True, 1.0, torch.float32),
        ("ASC-025-FP32", 2, 2*300, 8, 8, 64, 64, False, 1.0, torch.float32),
        ("ASC-026-FP32", 4, 4*512, 8, 8, 1, 1, False, 1.0, torch.float32),
        ("ASC-027-FP32", 4, 4*1024, 8, 8, 64, 64, True, 1.0, torch.float32),
        ("ASC-028-FP32", 4, 4*512, 8, 4, 64, 64, False, 1.0, torch.float32),
        ("ASC-029-FP32", 4, 4*512, 16, 4, 64, 64, False, 1.0, torch.float32),
        ("ASC-030-FP32", 8, 8*4096, 64, 64, 128, 128, True, 1.0, torch.float32),
        ("ASC-031-FP32", 4, 4*128, 16, 8, 31, 31, True, 1.0, torch.float32),
        ("ASC-032-FP32", 8, 8*128, 2, 2, 16, 16, True, 1.0, torch.float32),
        ("ASC-033-FP32", 1, 128, 3, 3, 128, 128, True, 1.0, torch.float32),
        ("ASC-034-FP32", 4, 4*2000, 2, 2, 16, 16, True, 1.0, torch.float32)
    ]
    return test_cases