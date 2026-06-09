import torch
import torch_npu
import math

def gen_inputs(seqlens_list_array, seqlens_list_kv_array, isTnd):
    B = 1
    NQuery = 64
    NQueryIndex = 64
    N2 = 1
    S1 = 128
    S2 = 128
    topK = 2048
    D = 512
    DIndex = 128
    DR = 64
    output_dtype = torch.float16
    q = torch.randn(B, S1, NQuery, D, dtype=output_dtype, device=torch.device('npu'))
    k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))

    q_index = torch.randn(B, S1, NQueryIndex, DIndex, dtype=output_dtype, device=torch.device('npu'))
    k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
    if DR != 0:
        q_rope = torch.randn(B, S1, NQuery, DR, dtype=output_dtype, device=torch.device('npu'))
        k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
    else:
        q_rope = None
        k_rope = None
    weights = torch.randn(B, S1, NQueryIndex, dtype=output_dtype, device=torch.device('npu'))
    a = -0.05  # 最小值
    b = 0.05 # 最大值
    kk = 3.0  # 控制分布范围（3σ 覆盖绝大多数值）
    scale = (b - a) / (2 * kk)
    shift = (a + b) / 2
    weights = weights * scale + shift
    if isTnd:
        sparse_indices = torch.zeros(S1, N2, topK).to(torch.int32).npu()
        tIdx = 0
        for bIdx in range(B):
            for s1Idx in range(seqlens_list_array[bIdx]):
                s2RealSize = (int)((seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx]) + s1Idx + 1)
                if s2RealSize <= 0:
                    s2RealSize = seqlens_list_kv_array[bIdx]
                
                if s2RealSize > topK:
                    s2RealLen = topK
                else:
                    s2RealLen = s2RealSize
                #处理S2无效行场景，把对应的sparse indices置为-1
                sparse_indices[tIdx, :, 0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                sparse_indices[tIdx, :, s2RealLen : topK] = -1
                tIdx = tIdx + 1
        q_tnd = q.squeeze(dim=0)
        k_tnd = k.squeeze(dim=0)
        q_index_tnd = q_index.squeeze(dim=0)
        k_index_tnd = k_index.squeeze(dim=0)
        if q_rope is not None:
            q_rope_tnd = q_rope.squeeze(dim=0)
            k_rope_tnd = k_rope.squeeze(dim=0)
        else :
            q_rope_tnd = None
            k_rope_tnd = None
        weights_tnd = weights.squeeze(dim=0)
        
        softmax_max = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
        softmax_sum = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
        return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, sparse_indices, softmax_max, softmax_sum
    else :
        sparse_indices = torch.zeros(B, S1, N2, topK).to(torch.int32).npu()
        for s1Idx in range(S1):
            s2RealSize = (int)(S2 - S1 + s1Idx + 1)
            if s2RealSize <= 0:
                s2RealSize = S2

            if s2RealSize > topK:
                s2RealLen = topK
            else:
                s2RealLen = s2RealSize
            sparse_indices[:, s1Idx, 0,  0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
            sparse_indices[:, s1Idx, 0,  s2RealLen : topK] = -1

        softmax_max = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
        softmax_sum = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
        return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum


actual_seq_qlen = [128]
actual_seq_kvlen = [128]
input_layout = 'TND'
isTnd = True
sparse_mode = 3
scale = 1.0 / math.sqrt(512)
q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(actual_seq_qlen, actual_seq_kvlen, isTnd)

d_qi, d_ki, dw, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
        query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_kvlen, layout=input_layout, sparse_mode=sparse_mode,
        pre_tokens=65536, next_tokens=65536)
torch.npu.synchronize()

print(f"\n=== dQueryIndex ===")
print(f"  shape: {d_qi.shape}, dtype: {d_qi.dtype}")
print(f"  range: [{d_qi.float().min().item():.6f}, {d_qi.float().max().item():.6f}]")
print(f"  first 10: {d_qi.float().flatten()[:10].tolist()}")

print(f"\n=== dKeyIndex ===")
print(f"  shape: {d_ki.shape}, dtype: {d_ki.dtype}")
print(f"  range: [{d_ki.float().min().item():.6f}, {d_ki.float().max().item():.6f}]")
print(f"  first 10: {d_ki.float().flatten()[:10].tolist()}")

print(f"\n=== dWeight ===")
print(f"  shape: {dw.shape}, dtype: {dw.dtype}")
print(f"  range: [{dw.float().min().item():.6f}, {dw.float().max().item():.6f}]")
print(f"  first 10: {dw.float().flatten()[:10].tolist()}")

print(f"\n=== Loss ===")
print(f"  value: {loss.float().item():.6f}")
