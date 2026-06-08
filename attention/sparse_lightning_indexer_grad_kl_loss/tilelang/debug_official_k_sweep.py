#!/usr/bin/env python3
"""
调试官方算子支持的参数范围
基于官方文档示例的参数构造方式，系统性测试不同K/S1/S2组合
"""

import torch
import torch_npu
import math

K_VALUES = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
DQUERY = 512
DQUERY_INDEX = 128
D_ROPE = 64


def gen_sparse_indices_causal(B, S1, S2, N2, topk):
    """对齐官方示例的causal模式sparse_indices构造"""
    sparse_indices = torch.zeros(B, S1, N2, topk).to(torch.int32).npu()
    for s1Idx in range(S1):
        s2RealSize = S2 - S1 + s1Idx + 1
        if s2RealSize <= 0:
            s2RealSize = S2
        if s2RealSize > topk:
            s2RealLen = topk
        else:
            s2RealLen = s2RealSize
        sparse_indices[:, s1Idx, 0, 0:s2RealLen] = (
            torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)
        ).npu()
        sparse_indices[:, s1Idx, 0, s2RealLen:topk] = -1
    return sparse_indices


def test_official_example_style(N1, Nidx1, S1, S2, topk, layout="BSND"):
    """对齐官方文档示例的完整构造方式"""
    B = 1
    N2 = 1
    G = N1 // N2
    dtype = torch.float16

    query = torch.randn(B, S1, N1, DQUERY, dtype=dtype).npu()
    key = torch.randn(B, S2, N2, DQUERY, dtype=dtype).npu()
    query_index = torch.randn(B, S1, Nidx1, DQUERY_INDEX, dtype=dtype).npu()
    key_index = torch.randn(B, S2, N2, DQUERY_INDEX, dtype=dtype).npu()
    query_rope = torch.randn(B, S1, N1, D_ROPE, dtype=dtype).npu()
    key_rope = torch.randn(B, S2, N2, D_ROPE, dtype=dtype).npu()

    a, b, kk = -0.05, 0.05, 3.0
    scale_w = (b - a) / (2 * kk)
    shift_w = (a + b) / 2
    weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    weights = weights * scale_w + shift_w

    sparse_indices = gen_sparse_indices_causal(B, S1, S2, N2, topk)
    softmax_max = torch.randn(B, N2, S1, G, dtype=torch.float32).npu()
    softmax_sum = torch.randn(B, N2, S1, G, dtype=torch.float32).npu()

    scale_value = 1.0 / math.sqrt(DQUERY)

    if layout == "TND":
        query_in = query.squeeze(0)
        key_in = key.squeeze(0)
        query_index_in = query_index.squeeze(0)
        key_index_in = key_index.squeeze(0)
        weights_in = weights.squeeze(0)
        sparse_indices_in = sparse_indices.squeeze(0)
        softmax_max_in = softmax_max.squeeze(0)
        softmax_sum_in = softmax_sum.squeeze(0)
        query_rope_in = query_rope.squeeze(0)
        key_rope_in = key_rope.squeeze(0)
        actual_seq_qlen = [S1]
        actual_seq_klen = [S2]
    else:
        query_in = query
        key_in = key
        query_index_in = query_index
        key_index_in = key_index
        weights_in = weights
        sparse_indices_in = sparse_indices
        softmax_max_in = softmax_max
        softmax_sum_in = softmax_sum
        query_rope_in = query_rope
        key_rope_in = key_rope
        actual_seq_qlen = [S1]
        actual_seq_klen = [S2]

    try:
        d_qi, d_ki, dw, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query_in, key_in, query_index_in, key_index_in, weights_in, sparse_indices_in,
            softmax_max_in, softmax_sum_in, scale_value,
            query_rope=query_rope_in,
            key_rope=key_rope_in,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=3,
            pre_tokens=65536,
            next_tokens=65536,
        )
        torch.npu.synchronize()
        return True, None
    except Exception as e:
        return False, str(e)[:100]


if __name__ == "__main__":
    print("=" * 70)
    print("调试官方算子支持的参数范围 (对齐官方文档示例)")
    print("=" * 70)

    # [1] 官方示例默认参数: N1=64, Nidx1=64, S1=128, S2=128, K=2048
    print("\n[1] 官方示例默认参数: N1=64, S1=128, S2=128, 变化K, BSND")
    for k in K_VALUES:
        ok, err = test_official_example_style(64, 64, 128, 128, k, "BSND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}: {status}" + (f" - {err}" if err else ""))

    # [2] S1=128, S2=K (S2随K变化)
    print("\n[2] N1=64, S1=128, S2=K, 变化K, BSND")
    for k in K_VALUES:
        ok, err = test_official_example_style(64, 64, 128, k, k, "BSND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}, S2={k}: {status}" + (f" - {err}" if err else ""))

    # [3] N1=128, S1=128, S2=128, 变化K
    print("\n[3] N1=128, S1=128, S2=128, 变化K, BSND")
    for k in K_VALUES:
        ok, err = test_official_example_style(128, 64, 128, 128, k, "BSND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}: {status}" + (f" - {err}" if err else ""))

    # [4] N1=128, S1=1, S2=K (我们当前的配置)
    print("\n[4] N1=128, S1=1, S2=K (当前precision_test配置), BSND")
    for k in K_VALUES:
        ok, err = test_official_example_style(128, 64, 1, k, k, "BSND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}, S2={k}: {status}" + (f" - {err}" if err else ""))

    # [5] N1=128, S1=128, S2=K (S1=128而非1)
    print("\n[5] N1=128, S1=128, S2=K, 变化K, BSND")
    for k in K_VALUES:
        ok, err = test_official_example_style(128, 64, 128, k, k, "BSND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}, S2={k}: {status}" + (f" - {err}" if err else ""))

    # [6] TND layout测试
    print("\n[6] N1=64, S1=128, S2=128, 变化K, TND")
    for k in K_VALUES:
        ok, err = test_official_example_style(64, 64, 128, 128, k, "TND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}: {status}" + (f" - {err}" if err else ""))

    # [7] S1=1的TND测试
    print("\n[7] N1=128, S1=1, S2=K, 变化K, TND")
    for k in K_VALUES:
        ok, err = test_official_example_style(128, 64, 1, k, k, "TND")
        status = "OK" if ok else "FAIL"
        print(f"  K={k}, S2={k}: {status}" + (f" - {err}" if err else ""))

    print("\n" + "=" * 70)
