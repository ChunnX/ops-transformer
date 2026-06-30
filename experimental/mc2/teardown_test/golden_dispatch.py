#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golden_dispatch.py —— DispatchTeardown(mxfp8) 输入造数 + CPU golden。
用法: golden_dispatch.py <bs> <topk> <h> <E> <R> <datadir> [seed]
mxfp8：token = fp8(e4m3fn,1B) payload + e8m0(1B) scale；group=32 → scaleH=h/32。dispatch 纯字节重排(无算术)。
造（写 datadir）:
  token_data.bin  fp8   [R*E*bs, h]       payload(块序 le*R+src，expert-major，每块 bs 槽带空洞)
  token_scale.bin u8    [R*E*bs, h/32]    scale(e8m0 字节，与 payload 同块结构)，取值 [120,134]→2^-7..2^7
  wish_cnt.bin    int32 [R*E]             每块实收(expert-major: idx=le*R+src)
  expert_ids.bin  int32 [bs, topk]        每行 topk 个互不相同全局专家
golden:
  expand_x.golden.bin / expand_scale.golden.bin / send_counts.golden.bin / expand_idx.golden.bin / expert_token_nums.golden.bin
"""
import sys, os
import numpy as np
import ml_dtypes
FP8 = ml_dtypes.float8_e4m3fn


def main():
    bs, topk, h, E, R = (int(sys.argv[i]) for i in range(1, 6))
    datadir = sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1234
    os.makedirs(datadir, exist_ok=True)
    rng = np.random.default_rng(seed)
    GE = R * E
    minKE = min(topk, E)
    A = R * bs * minKE
    scaleH = h // 32

    # ---- 输入 ----
    expert_ids = np.zeros((bs, topk), dtype=np.int32)
    for i in range(bs):
        expert_ids[i] = rng.choice(GE, size=topk, replace=False).astype(np.int32)
    # wishCnt 按真实路由模拟（非独立随机）：每源卡 src 的 bs 个 token 各选 topk 个全局专家，
    # 统计落到本卡 E 个本地专家(全局 [0,E))的数量 → wishCnt[le*R+src]。
    # 天然保证 per-block ≤ bs 且 Σ ≤ R*bs*min(topk,E)=A（每 token 命中本卡 ≤ min(topk,E) 个专家）。
    wish_cnt = np.zeros(GE, dtype=np.int32)
    for src in range(R):
        for _ in range(bs):
            picks = set(int(x) for x in rng.choice(GE, size=topk, replace=False))
            for le in range(E):
                if le in picks:
                    wish_cnt[le * R + src] += 1
    token_data = rng.standard_normal((GE * bs, h)).astype(np.float32).astype(FP8)        # fp8 payload
    token_scale = rng.integers(120, 135, size=(GE * bs, scaleH)).astype(np.uint8)        # e8m0 字节

    token_data.tofile(os.path.join(datadir, "token_data.bin"))
    token_scale.tofile(os.path.join(datadir, "token_scale.bin"))
    wish_cnt.tofile(os.path.join(datadir, "wish_cnt.bin"))
    expert_ids.tofile(os.path.join(datadir, "expert_ids.bin"))

    # ---- golden: expand_idx 出现次数 ----
    expand_idx = np.zeros((bs, topk), dtype=np.int32)
    counter = np.zeros(GE, dtype=np.int32)
    for i in range(bs):
        for j in range(topk):
            ge = int(expert_ids[i, j])
            expand_idx[i, j] = counter[ge]
            counter[ge] += 1

    # 数据区与 expandX 同为 expert-major（块序 le*R+src）；wishCnt 与块一一对应、同序。
    # ---- golden: expert_token_nums[le] = Σ_src wishCnt[le*R+src] ----
    expert_token_nums = np.zeros(E, dtype=np.int32)
    for le in range(E):
        expert_token_nums[le] = sum(int(wish_cnt[le * R + src]) for src in range(R))

    # ---- golden: send_counts inclusive 前缀和（wishCnt 自然序即 expert-major）----
    send_counts = np.zeros(GE, dtype=np.int32)
    run = 0
    for t in range(GE):
        run += int(wish_cnt[t])
        send_counts[t] = run

    # ---- golden: expand_x / expand_scale 去空洞紧密化（块 t 源即数据区块 t，payload+scale 锁步）----
    expand_x = np.zeros((A, h), dtype=FP8)
    expand_scale = np.zeros((A, scaleH), dtype=np.uint8)
    dst = 0
    for t in range(GE):
        cnt = int(wish_cnt[t])
        for r in range(cnt):
            expand_x[dst] = token_data[t * bs + r]
            expand_scale[dst] = token_scale[t * bs + r]
            dst += 1
    assert dst <= A, f"dst={dst} > A={A}"

    expand_x.tofile(os.path.join(datadir, "expand_x.golden.bin"))
    expand_scale.tofile(os.path.join(datadir, "expand_scale.golden.bin"))
    send_counts.tofile(os.path.join(datadir, "send_counts.golden.bin"))
    expand_idx.tofile(os.path.join(datadir, "expand_idx.golden.bin"))
    expert_token_nums.tofile(os.path.join(datadir, "expert_token_nums.golden.bin"))
    print(f"[golden] bs={bs} topk={topk} h={h} E={E} R={R} GE={GE} A={A} scaleH={scaleH} valid_rows={dst} seed={seed} (mxfp8)")
    print(f"[golden] wish_cnt(expert-major le*R+src)={wish_cnt.tolist()}")
    print(f"[golden] send_counts(expert-major)={send_counts.tolist()}")


if __name__ == "__main__":
    main()
