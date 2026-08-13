import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class FusedInferAttentionScoreGolden:
    """FusedInferAttentionScore Golden reference implementation (non-quantized).

    Supports two computation modes:
      - "full":    per-batch bmm(Q,K) -> softmax -> bmm(P,V)
      - "tiled":   block-wise online softmax, mirroring operator internals
                   Q_BLOCK=128, K_BLOCK=256, V_BLOCK=512

    Both modes support device_mode="cpu" and "npu".
    """

    TILE_SIZE = 2048
    EPSILON = 1e-20
    TILED_CROSSOVER = 3 * 1024 * 1024  # S1*S2 > this -> tiled saves memory

    def __init__(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int = 128,
        softmax_scale: Optional[float] = None,
        tile_size: Optional[int] = None,
    ):
        self._num_heads_q = num_heads_q
        self._num_heads_kv = num_heads_kv
        self._head_dim = head_dim
        self._softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        self._tile_size = tile_size or self.TILE_SIZE

    # ==========================================================================
    # Public entry points
    # ==========================================================================

    def forward_golden(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_rope: Optional[torch.Tensor] = None,
        key_rope: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sparse_mode: int = 0,
        atten_mask: Optional[torch.Tensor] = None,
        layout_q: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
        return_softmax_lse: int = 0,
        device_mode: str = "cpu",
        compute_mode: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fused_infer_attention_score golden reference.

        Args:
            query_rope: RoPE part of Q, concatenated to Q along last dim (MLA).
            key_rope: RoPE part of K, concatenated to K along last dim (MLA).
            compute_mode: "auto" selects based on S1*S2 crossover,
                          "tiled" for block-wise online softmax,
                          "full" for per-batch full bmm.

        Returns:
            (attn_out, softmax_lse)
            attn_out: same dtype as q, shape determined by layout_out.
            softmax_lse: float32, shape (1,) if return_softmax_lse=0.
        """
        if device_mode not in ("cpu", "npu"):
            raise ValueError(f"device_mode must be 'cpu' or 'npu', got {device_mode}")
        if compute_mode not in ("auto", "tiled", "full"):
            raise ValueError(f"compute_mode must be 'auto', 'tiled' or 'full', got {compute_mode}")

        if device_mode == "npu":
            for name, t in [("q", q), ("k", k), ("v", v)]:
                if t.device.type != "npu":
                    raise ValueError(
                        f"device_mode='npu' requires {name} to be on NPU, "
                        f"got {t.device}"
                    )

        orig_dtype = q.dtype
        scale = softmax_scale if softmax_scale is not None else self._softmax_scale

        # 1. For cpu mode, move auxiliary tensors to cpu (q/k/v stay as-is)
        if device_mode == "cpu":
            if block_table is not None:
                block_table = block_table.cpu()
            if cu_seqlens_q is not None:
                cu_seqlens_q = cu_seqlens_q.cpu()
            if cu_seqlens_kv is not None:
                cu_seqlens_kv = cu_seqlens_kv.cpu()
            if seqused_q is not None:
                seqused_q = seqused_q.cpu()
            if seqused_kv is not None:
                seqused_kv = seqused_kv.cpu()
            if atten_mask is not None:
                atten_mask = atten_mask.cpu()
            if sinks is not None:
                sinks = sinks.cpu()

        # 1.5. Concatenate RoPE to Q (MLA support)
        # K rope is handled per-batch in _extract_batch_bnsd_f32 (PA blocks)
        if query_rope is not None and key_rope is not None:
            if device_mode == "cpu":
                query_rope = query_rope.cpu()
                key_rope = key_rope.cpu()
            q = torch.cat([q, query_rope], dim=-1)

        # 2. Compute per-batch actual seq lengths
        if seqused_q is not None:
            q_lens = seqused_q.to(torch.int32).tolist()
        elif cu_seqlens_q is not None:
            cu_q = cu_seqlens_q.to(torch.int32).tolist()
            q_lens = [cu_q[b + 1] - cu_q[b] for b in range(len(cu_q) - 1)]
        else:
            q_lens = None

        if seqused_kv is not None:
            kv_lens = seqused_kv.to(torch.int32).tolist()
        elif cu_seqlens_kv is not None:
            cu_kv = cu_seqlens_kv.to(torch.int32).tolist()
            kv_lens = [cu_kv[b + 1] - cu_kv[b] for b in range(len(cu_kv) - 1)]
        else:
            kv_lens = None

        # Resolve auto compute_mode: tiled saves memory only when S1*S2 is large
        # and S1 >= TILE_SIZE (otherwise single partial Q tile can't amortize overhead)
        if compute_mode == "auto":
            if q_lens is not None and kv_lens is not None:
                max_sq = max(q_lens)
                max_product = max(sq * skv for sq, skv in zip(q_lens, kv_lens))
            else:
                max_sq = q.shape[2] if layout_q == "BNSD" else q.shape[1]
                max_skv = k.shape[2] if layout_kv == "BNSD" else k.shape[1]
                max_product = max_sq * max_skv
            compute_mode = "tiled" if (max_sq >= self._tile_size and
                                       max_product > self.TILED_CROSSOVER) else "full"

        # 3. Compute attention — per-batch to avoid OOM on large TND inputs
        out_f32, lse_f32 = self._compute_per_batch(
            q, k, v, q_lens, kv_lens,
            layout_q, layout_kv, cu_seqlens_q, cu_seqlens_kv,
            block_table, seqused_kv,
            sparse_mode, atten_mask, scale, sinks, compute_mode,
            key_rope,
        )

        # 6. Convert output to target layout and dtype
        if layout_out == "TND":
            # _compute_per_batch already wrote TND directly
            attn_out = out_f32.to(orig_dtype)
        else:
            out_bnsd = out_f32.to(orig_dtype)
            if layout_out == "BNSD":
                attn_out = out_bnsd
            elif layout_out == "BSND":
                attn_out = out_bnsd.permute(0, 2, 1, 3).contiguous()
            else:
                raise ValueError(f"Unsupported layout_out: {layout_out}")

        # 7. Format softmax_lse
        if return_softmax_lse == 0:
            softmax_lse = torch.zeros(1, dtype=torch.float32, device=attn_out.device)
        else:
            if layout_out == "TND":
                # _compute_per_batch 输出 (T, N_q), NPU 算子 TND 约定 (N_q, T)
                softmax_lse = lse_f32.t().contiguous()
            else:
                softmax_lse = lse_f32  # (B, N_q, Sq)

        return attn_out, softmax_lse

    # ==========================================================================
    # Per-batch compute (avoids OOM by not converting all batches at once)
    # ==========================================================================

    def _compute_per_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[list],
        kv_lens: Optional[list],
        layout_q: str,
        layout_kv: str,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_kv: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        seqused_kv: Optional[torch.Tensor],
        sparse_mode: int,
        atten_mask: Optional[torch.Tensor],
        scale: float,
        sinks: Optional[torch.Tensor],
        compute_mode: str = "tiled",
        key_rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention per-batch to avoid OOM on large TND inputs.

        Each batch's q/k/v is converted to BNSD and cast to fp32 individually,
        then passed to _compute_tiled_single or _compute_full_single.
        """
        N_q = self._num_heads_q
        N_kv = self._num_heads_kv
        D = self._head_dim
        if layout_kv == "PA_NZ":
            dv = v.shape[2] * v.shape[4]
        elif layout_kv == "PA_BBH":
            dv = v.shape[-1] // N_kv
        else:
            dv = v.shape[-1]
        group = N_q // N_kv

        # Determine B and seq lengths
        if layout_q == "TND":
            cu_q = cu_seqlens_q.to(torch.int64).tolist()
            B = len(cu_q) - 1
        elif layout_q == "BNSD":
            B = q.shape[0]
        else:  # BSND
            B = q.shape[0]

        if q_lens is None:
            if layout_q == "TND":
                q_lens = [cu_q[b + 1] - cu_q[b] for b in range(B)]
            elif layout_q == "BNSD":
                q_lens = [q.shape[2]] * B
            else:  # BSND
                q_lens = [q.shape[1]] * B

        if kv_lens is None:
            if layout_kv == "TND" and cu_seqlens_kv is not None:
                cu_kv = cu_seqlens_kv.to(torch.int64).tolist()
                kv_lens = [cu_kv[b + 1] - cu_kv[b] for b in range(B)]
            elif layout_kv in ("PA_BBND", "PA_BNBD", "PA_BBH", "PA_NZ"):
                kv_lens = self._get_kv_lens(seqused_kv, cu_seqlens_kv, block_table)
            elif layout_kv == "BNSD":
                kv_lens = [k.shape[2]] * B
            else:  # BSND
                kv_lens = [k.shape[1]] * B

        # For TND output, write directly to TND to avoid large BNSD allocation
        is_tnd_out = (layout_q == "TND")
        if is_tnd_out:
            # TND 输出 T 与输入 q 行数一致 (cu_seqlens_q[-1]), 含 padding 行;
            # 写入偏移用 cu_q[b], 末尾 batch 不会越界。
            total_q = q.shape[0]
            out_tnd = torch.zeros(total_q, N_q, dv, dtype=torch.float32, device=q.device)
            lse_tnd = torch.zeros(total_q, N_q, dtype=torch.float32, device=q.device)
            out = None
            lse = None
        else:
            max_sq = max(q_lens) if q_lens else 0
            out = torch.zeros(B, N_q, max_sq, dv, dtype=torch.float32, device=q.device)
            lse = torch.zeros(B, N_q, max_sq, dtype=torch.float32, device=q.device)
            out_tnd = None
            lse_tnd = None

        for b in range(B):
            sq = q_lens[b]
            skv = kv_lens[b]
            if sq <= 0 or skv <= 0:
                continue

            # Extract per-batch q/k/v and convert to BNSD fp32
            q_b, k_b, v_b = self._extract_batch_bnsd_f32(
                q, k, v, b, sq, skv, layout_q, layout_kv,
                cu_seqlens_q, cu_seqlens_kv, block_table, seqused_kv,
                key_rope,
            )

            # Build mask for this batch
            mask_b = self._build_batch_mask(sq, skv, sparse_mode, atten_mask, q.device)

            # Compute per head
            compute_fn = self._compute_tiled_single if compute_mode == "tiled" else self._compute_full_single
            delta = skv - sq
            for kv_h in range(N_kv):
                k_bh = k_b[0, kv_h, :skv, :]
                v_bh = v_b[0, kv_h, :skv, :]
                for g in range(group):
                    qh = kv_h * group + g
                    q_bh = q_b[0, qh, :sq, :]
                    sink = sinks[qh] if sinks is not None else None
                    if compute_mode == "tiled":
                        o, l = compute_fn(q_bh, k_bh, v_bh, mask_b, scale, sink, sparse_mode, delta)
                    else:
                        o, l = compute_fn(q_bh, k_bh, v_bh, mask_b, scale, sink)
                    if is_tnd_out:
                        q_start = cu_q[b]
                        out_tnd[q_start:q_start + sq, qh, :] = o
                        lse_tnd[q_start:q_start + sq, qh] = l
                    else:
                        out[b, qh, :sq, :] = o
                        lse[b, qh, :sq] = l

            del q_b, k_b, v_b

        if is_tnd_out:
            return out_tnd, lse_tnd
        return out, lse

    @staticmethod
    def _extract_batch_bnsd_f32(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: int,
        sq: int,
        skv: int,
        layout_q: str,
        layout_kv: str,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_kv: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        seqused_kv: Optional[torch.Tensor],
        key_rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract single batch q/k/v as (1, N, S, D) float32.
        If key_rope is provided, it is concatenated to K along the last dim (MLA).
        """
        # Q
        if layout_q == "BNSD":
            q_b = q[b:b+1, :, :sq, :].float()
        elif layout_q == "BSND":
            q_b = q[b:b+1, :sq, :, :].permute(0, 2, 1, 3).contiguous().float()
        elif layout_q == "TND":
            cu_q = cu_seqlens_q.to(torch.int64).tolist()
            s, e = cu_q[b], cu_q[b + 1]
            q_b = q[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
        else:
            raise ValueError(f"Unsupported layout_q: {layout_q}")

        # K, V
        if layout_kv == "BNSD":
            k_b = k[b:b+1, :, :skv, :].float()
            v_b = v[b:b+1, :, :skv, :].float()
        elif layout_kv == "BSND":
            k_b = k[b:b+1, :skv, :, :].permute(0, 2, 1, 3).contiguous().float()
            v_b = v[b:b+1, :skv, :, :].permute(0, 2, 1, 3).contiguous().float()
        elif layout_kv == "TND":
            cu_kv = cu_seqlens_kv.to(torch.int64).tolist()
            s, e = cu_kv[b], cu_kv[b + 1]
            k_b = k[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
            v_b = v[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
        elif layout_kv in ("PA_BBND", "PA_BNBD"):
            N_kv = k.shape[1] if layout_kv == "PA_BNBD" else k.shape[2]
            D = k.shape[-1]
            dv = v.shape[-1]
            bs = k.shape[2] if layout_kv == "PA_BNBD" else k.shape[1]
            k_b = torch.zeros(1, N_kv, skv, D, dtype=torch.float32, device=k.device)
            v_b = torch.zeros(1, N_kv, skv, dv, dtype=torch.float32, device=v.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                if layout_kv == "PA_BBND":
                    k_b[0, :, offset:offset + valid, :] = k[block_id, :valid, :, :].permute(1, 0, 2).float()
                    v_b[0, :, offset:offset + valid, :] = v[block_id, :valid, :, :].permute(1, 0, 2).float()
                else:
                    k_b[0, :, offset:offset + valid, :] = k[block_id, :, :valid, :].float()
                    v_b[0, :, offset:offset + valid, :] = v[block_id, :, :valid, :].float()
        elif layout_kv == "PA_BBH":
            N_kv = 1
            D = k.shape[-1]
            dv = v.shape[-1]
            bs = k.shape[1]
            k_b = torch.zeros(1, N_kv, skv, D, dtype=torch.float32, device=k.device)
            v_b = torch.zeros(1, N_kv, skv, dv, dtype=torch.float32, device=v.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                k_b[0, 0, offset:offset + valid, :] = k[block_id, :valid, :].float()
                v_b[0, 0, offset:offset + valid, :] = v[block_id, :valid, :].float()
        elif layout_kv == "PA_NZ":
            N_kv = k.shape[1]
            D = k.shape[2] * k.shape[4]
            dv = v.shape[2] * v.shape[4]
            bs = k.shape[3]
            k_b = torch.zeros(1, N_kv, skv, D, dtype=torch.float32, device=k.device)
            v_b = torch.zeros(1, N_kv, skv, dv, dtype=torch.float32, device=v.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                k_blk = k[block_id, :, :, :valid, :].permute(0, 2, 1, 3).contiguous().reshape(N_kv, valid, D)
                v_blk = v[block_id, :, :, :valid, :].permute(0, 2, 1, 3).contiguous().reshape(N_kv, valid, dv)
                k_b[0, :, offset:offset + valid, :] = k_blk.float()
                v_b[0, :, offset:offset + valid, :] = v_blk.float()
        else:
            raise ValueError(f"Unsupported layout_kv: {layout_kv}")

        # Concatenate key_rope to K per-batch (MLA support)
        if key_rope is not None:
            kr_b = FusedInferAttentionScoreGolden._extract_key_rope_batch(
                key_rope, b, skv, layout_kv, block_table,
            )
            k_b = torch.cat([k_b, kr_b], dim=-1)

        return q_b, k_b, v_b

    @staticmethod
    def _extract_key_rope_batch(
        key_rope: torch.Tensor,
        b: int,
        skv: int,
        layout_kv: str,
        block_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract single batch key_rope as (1, N_kv, S, rope_d) float32."""
        if layout_kv in ("BNSD", "BSND", "TND"):
            return FusedInferAttentionScoreGolden._extract_batch_bnsd_f32(
                key_rope, key_rope, key_rope, b, skv, skv, layout_kv, layout_kv,
                None, None, None, None,
            )[1]

        if layout_kv in ("PA_BBND", "PA_BNBD"):
            N_kv = key_rope.shape[1] if layout_kv == "PA_BNBD" else key_rope.shape[2]
            rope_d = key_rope.shape[-1]
            bs = key_rope.shape[2] if layout_kv == "PA_BNBD" else key_rope.shape[1]
            kr_b = torch.zeros(1, N_kv, skv, rope_d, dtype=torch.float32, device=key_rope.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                if layout_kv == "PA_BBND":
                    kr_b[0, :, offset:offset + valid, :] = key_rope[block_id, :valid, :, :].permute(1, 0, 2).float()
                else:
                    kr_b[0, :, offset:offset + valid, :] = key_rope[block_id, :, :valid, :].float()
            return kr_b

        if layout_kv == "PA_BBH":
            rope_d = key_rope.shape[-1]
            bs = key_rope.shape[1]
            kr_b = torch.zeros(1, 1, skv, rope_d, dtype=torch.float32, device=key_rope.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                kr_b[0, 0, offset:offset + valid, :] = key_rope[block_id, :valid, :].float()
            return kr_b

        if layout_kv == "PA_NZ":
            N_kv = key_rope.shape[1]
            rope_d = key_rope.shape[2] * key_rope.shape[4]
            bs = key_rope.shape[3]
            kr_b = torch.zeros(1, N_kv, skv, rope_d, dtype=torch.float32, device=key_rope.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                kr_blk = key_rope[block_id, :, :, :valid, :].permute(0, 2, 1, 3).contiguous().reshape(N_kv, valid, rope_d)
                kr_b[0, :, offset:offset + valid, :] = kr_blk.float()
            return kr_b

        raise ValueError(f"Unsupported layout_kv for key_rope: {layout_kv}")

    # ==========================================================================
    # Tiled (segmented) golden - block-wise online softmax
    # ==========================================================================

    def _compute_tiled_single(
        self,
        q: torch.Tensor,       # (Sq, D) float32
        k: torch.Tensor,       # (Skv, D) float32
        v: torch.Tensor,       # (Skv, D) float32
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool, True=mask out
        scale: float,
        sink: Optional[float],
        sparse_mode: int = 0,
        delta: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tiled online softmax for a single (batch, head) pair.

        Scores matrix Q@K^T of shape (Sq, Skv) is processed in square
        tiles of size (TILE, TILE). K/V blocks are 1:1 (no pairing).

        When mask is None but sparse_mode=3, causal masking is applied
        per-tile to avoid materializing the full (Sq, Skv) mask.

        Returns (out, lse) where out is (Sq, dv) and lse is (Sq,).
        """
        Sq, D = q.shape
        Skv = k.shape[0]
        dv = v.shape[-1]
        BS = self._tile_size

        TILES_Q = (Sq + BS - 1) // BS
        TILES_KV = (Skv + BS - 1) // BS

        out = torch.zeros(Sq, dv, dtype=torch.float32, device=q.device)
        o_sum = torch.zeros(Sq, 1, dtype=torch.float32, device=q.device)
        o_max = torch.full(
            (Sq, 1), torch.finfo(torch.float32).min,
            dtype=torch.float32, device=q.device,
        )

        Q_BLOCKS = list(torch.split(q, BS, dim=0))
        K_BLOCKS = list(torch.split(k, BS, dim=0))
        V_BLOCKS = list(torch.split(v, BS, dim=0))
        o_BLOCKS = list(torch.split(out, BS, dim=0))
        s_BLOCKS = list(torch.split(o_sum, BS, dim=0))
        m_BLOCKS = list(torch.split(o_max, BS, dim=0))

        for i in range(TILES_Q):
            Qi = Q_BLOCKS[i]  # (Bq, D)
            Sq_start = i * BS
            Sq_end = min(Sq_start + BS, Sq)
            Bq = Sq_end - Sq_start

            for j in range(TILES_KV):
                oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

                Kj = K_BLOCKS[j]   # (Bk, D)
                Vj = V_BLOCKS[j]   # (Bv, D)
                Sk_start = j * BS
                Sk_end = min(Sk_start + BS, Skv)
                Bk = Sk_end - Sk_start

                # Score tile: (Bq, Bk) — a square block from Q@K^T
                S_ij = torch.matmul(Qi, Kj.t()) * scale

                # Apply mask
                if mask is not None:
                    S_ij = S_ij.masked_fill(
                        mask[Sq_start:Sq_end, Sk_start:Sk_end], float("-inf")
                    )
                elif sparse_mode == 3:
                    # Per-tile causal: k_idx > q_idx + delta
                    q_idx = torch.arange(Sq_start, Sq_end, device=q.device).unsqueeze(1)
                    k_idx = torch.arange(Sk_start, Sk_end, device=q.device).unsqueeze(0)
                    tile_mask = k_idx > (q_idx + delta)
                    S_ij = S_ij.masked_fill(tile_mask, float("-inf"))

                # Online softmax update
                m_block = S_ij.amax(dim=-1, keepdim=True)
                m_new = torch.maximum(mi, m_block)

                P_ij = torch.exp(S_ij - m_new)
                s_block = P_ij.sum(dim=-1, keepdim=True)

                # Rescale old accumulators and accumulate new P@V
                update_old = torch.exp(mi - m_new)
                P_V = torch.matmul(P_ij, Vj[:Kj.shape[0], :])

                o_BLOCKS[i] = update_old * oi + P_V
                s_BLOCKS[i] = update_old * si + s_block
                m_BLOCKS[i] = m_new

        out = torch.cat(o_BLOCKS, dim=0)
        out_sum = torch.cat(s_BLOCKS, dim=0)
        out = out / (out_sum + self.EPSILON)

        out_max = torch.cat(m_BLOCKS, dim=0)
        lse = torch.log(out_sum + self.EPSILON) + out_max
        lse = lse.squeeze(-1)  # (Sq,)

        return out, lse

    # ==========================================================================
    # Full (non-tiled) golden - per-batch bmm
    # ==========================================================================

    def _compute_full_single(
        self,
        q: torch.Tensor,       # (Sq, D) float32
        k: torch.Tensor,       # (Skv, D) float32
        v: torch.Tensor,       # (Skv, D) float32
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool
        scale: float,
        sink: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full attention for a single (batch, head) pair.

        Returns (out, lse) where out is (Sq, dv) and lse is (Sq,).
        """
        scores = torch.matmul(q, k.t()) * scale  # (Sq, Skv)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        row_max = scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )

        if sink is not None:
            row_max = torch.maximum(row_max, torch.full_like(row_max, sink))

        exp_scores = torch.exp(scores - row_max)
        row_sum = exp_scores.sum(dim=-1, keepdim=True)

        if sink is not None:
            row_sum = row_sum + math.exp(sink - row_max)

        probs = exp_scores / row_sum
        out = torch.matmul(probs, v)

        lse = torch.log(row_sum + self.EPSILON) + row_max
        lse = lse.squeeze(-1)

        return out, lse

    # ==========================================================================
    # Mask generation
    # ==========================================================================

    def _build_batch_mask(
        self,
        sq: int,
        skv: int,
        sparse_mode: int,
        atten_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build (sq, skv) bool mask for a single batch. True = mask out.

        For large sq*skv, this would OOM. Instead return None and let
        _compute_tiled_single handle masking per-tile.
        """
        if sparse_mode == 0:
            return None

        # For large matrices, skip full mask creation — tile-level masking
        # will be done inside _compute_tiled_single
        if sq * skv > 64 * 1024 * 1024:  # > 64M elements
            return None

        if sparse_mode == 3:
            delta = skv - sq
            q_idx = torch.arange(sq, device=device).unsqueeze(1)
            k_idx = torch.arange(skv, device=device).unsqueeze(0)
            return k_idx > (q_idx + delta)

        if atten_mask is not None:
            return atten_mask[:sq, :skv].to(torch.bool)

        return None

    # ==========================================================================
    # Layout conversion
    # ==========================================================================

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout_q: str,
        layout_kv: str,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_kv: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        seqused_kv: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert q/k/v to internal BNSD (B, N, S, D)."""
        # Q
        if layout_q == "BNSD":
            q_bnsd = q
        elif layout_q == "BSND":
            q_bnsd = q.permute(0, 2, 1, 3).contiguous()
        elif layout_q == "TND":
            q_bnsd = self._tnd_to_bnsd(q, cu_seqlens_q, self._num_heads_q, self._head_dim)
        else:
            raise ValueError(f"Unsupported layout_q: {layout_q}")

        # K, V
        if layout_kv in ("BNSD", "BSND"):
            if layout_kv == "BNSD":
                k_bnsd = k
                v_bnsd = v
            else:
                k_bnsd = k.permute(0, 2, 1, 3).contiguous()
                v_bnsd = v.permute(0, 2, 1, 3).contiguous()
        elif layout_kv == "TND":
            k_bnsd = self._tnd_to_bnsd(k, cu_seqlens_kv, self._num_heads_kv, self._head_dim)
            v_bnsd = self._tnd_to_bnsd(v, cu_seqlens_kv, self._num_heads_kv, self._head_dim)
        elif layout_kv in ("PA_BBND", "PA_BNBD"):
            kv_lens = self._get_kv_lens(seqused_kv, cu_seqlens_kv, block_table)
            k_bnsd = self._reconstruct_pa_kv(k, block_table, kv_lens, layout_kv)
            v_bnsd = self._reconstruct_pa_kv(v, block_table, kv_lens, layout_kv)
        else:
            raise ValueError(f"Unsupported layout_kv: {layout_kv}")

        return q_bnsd, k_bnsd, v_bnsd

    @staticmethod
    def _tnd_to_bnsd(
        tnd: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """(T, N, D) -> (B, N, max_S, D)."""
        cu = cu_seqlens.to(torch.int64).tolist()
        B = len(cu) - 1
        max_s = max(cu[b + 1] - cu[b] for b in range(B))
        bnsd = torch.zeros(
            B, num_heads, max_s, head_dim, dtype=tnd.dtype, device=tnd.device
        )
        for b in range(B):
            s_start = cu[b]
            s_end = cu[b + 1]
            s_len = s_end - s_start
            # tnd[s_start:s_end] is (s_len, N, D) -> permute to (N, s_len, D)
            bnsd[b, :, :s_len, :] = tnd[s_start:s_end].permute(1, 0, 2)
        return bnsd

    @staticmethod
    def _reconstruct_pa_kv(
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: list,
        kv_layout: str,
    ) -> torch.Tensor:
        """Reconstruct KV from paged cache to BNSD.

        PA_BBND: (num_blocks, block_size, KV_N, D)
        PA_BNBD: (num_blocks, KV_N, block_size, D)
        -> (B, KV_N, max_kv_s, D)
        """
        B = block_table.shape[0]
        if kv_layout == "PA_BBND":
            block_size = kv_cache.shape[1]
            KV_N = kv_cache.shape[2]
            D = kv_cache.shape[3]
        else:  # PA_BNBD
            block_size = kv_cache.shape[2]
            KV_N = kv_cache.shape[1]
            D = kv_cache.shape[3]

        max_kv_s = max(kv_lens)
        bnsd = torch.zeros(
            B, KV_N, max_kv_s, D, dtype=kv_cache.dtype, device=kv_cache.device
        )
        bt = block_table.to(torch.int64)
        for b in range(B):
            kv_len = kv_lens[b]
            num_blocks_b = (kv_len + block_size - 1) // block_size
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * block_size
                valid = min(block_size, kv_len - offset)
                if valid <= 0:
                    continue
                if kv_layout == "PA_BBND":
                    # kv_cache[block_id, :valid, :, :] is (valid, KV_N, D) -> permute to (KV_N, valid, D)
                    bnsd[b, :, offset:offset + valid, :] = kv_cache[block_id, :valid, :, :].permute(1, 0, 2)
                else:
                    # PA_BNBD: kv_cache[block_id, :, :valid, :] is already (KV_N, valid, D)
                    bnsd[b, :, offset:offset + valid, :] = kv_cache[block_id, :, :valid, :]
        return bnsd

    @staticmethod
    def _get_kv_lens(
        seqused_kv: Optional[torch.Tensor],
        cu_seqlens_kv: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
    ) -> list:
        if seqused_kv is not None:
            return seqused_kv.to(torch.int32).tolist()
        elif cu_seqlens_kv is not None:
            cu = cu_seqlens_kv.to(torch.int32).tolist()
            return [cu[b + 1] - cu[b] for b in range(len(cu) - 1)]
        elif block_table is not None:
            B = block_table.shape[0]
            raise ValueError(
                "PA mode requires seqused_kv or cu_seqlens_kv to determine kv lengths"
            )
        else:
            raise ValueError("Cannot determine kv lengths")

    @staticmethod
    def _broadcast_gqa(
        kv_bnsd: torch.Tensor,
        num_heads_q: int,
        num_heads_kv: int,
    ) -> torch.Tensor:
        """(B, KV_N, S, D) -> (B, Q_N, S, D) via GQA broadcast."""
        if num_heads_q == num_heads_kv:
            return kv_bnsd
        ratio = num_heads_q // num_heads_kv
        head_idx = torch.arange(num_heads_q, device=kv_bnsd.device) // ratio
        return kv_bnsd.index_select(1, head_idx).contiguous()

    @staticmethod
    def _bnsd_to_tnd(
        bnsd: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N, S, D) -> (T, N, D)."""
        cu = cu_seqlens_q.to(torch.int64).tolist()
        B, N, _, D = bnsd.shape
        T = cu[-1]
        tnd = torch.zeros(T, N, D, dtype=bnsd.dtype, device=bnsd.device)
        for b in range(B):
            s_start = cu[b]
            s_end = cu[b + 1]
            s_len = s_end - s_start
            # bnsd[b, :, :s_len, :] is (N, s_len, D) -> permute to (s_len, N, D)
            tnd[s_start:s_end] = bnsd[b, :, :s_len, :].permute(1, 0, 2)
        return tnd

    @staticmethod
    def _lse_bnsd_to_tnd(
        lse_bnsd: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N, S) -> (N, T)."""
        cu = cu_seqlens_q.to(torch.int64).tolist()
        B, N, S = lse_bnsd.shape
        T = cu[-1]
        lse_tnd = torch.zeros(N, T, dtype=torch.float32, device=lse_bnsd.device)
        for b in range(B):
            s_start = cu[b]
            s_end = cu[b + 1]
            s_len = s_end - s_start
            # lse_bnsd[b, :, :s_len] is (N, s_len) -> already matches (N, T) slice
            lse_tnd[:, s_start:s_end] = lse_bnsd[b, :, :s_len]
        return lse_tnd

    # ==========================================================================
    # Device helpers
    # ==========================================================================


class FusedInferAttentionScore(nn.Module):
    """FusedInferAttentionScore operator Module wrapper with golden interface."""

    def __init__(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int = 128,
        softmax_scale: Optional[float] = None,
        sparse_mode: int = 0,
        layout_q: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
    ):
        super().__init__()
        self._num_heads_q = num_heads_q
        self._num_heads_kv = num_heads_kv
        self._head_dim = head_dim
        self._softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        self._sparse_mode = sparse_mode
        self._layout_q = layout_q
        self._layout_kv = layout_kv
        self._layout_out = layout_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        atten_mask: Optional[torch.Tensor] = None,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        return_softmax_lse: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call NPU fused_infer_attention_score operator."""
        import cann_ops_transformer

        attn_out, softmax_lse = cann_ops_transformer.ops.flash_attn(
            q, k, v,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=self._softmax_scale,
            sparse_mode=self._sparse_mode,
            attn_mask=atten_mask,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=self._layout_q,
            layout_kv=self._layout_kv,
            layout_out=self._layout_out,
            return_softmax_lse=return_softmax_lse,
        )
        return attn_out, softmax_lse

    def forward_golden(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_rope: Optional[torch.Tensor] = None,
        key_rope: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sparse_mode: Optional[int] = None,
        atten_mask: Optional[torch.Tensor] = None,
        layout_q: Optional[str] = None,
        layout_kv: Optional[str] = None,
        layout_out: Optional[str] = None,
        return_softmax_lse: int = 0,
        device_mode: str = "cpu",
        compute_mode: str = "auto",
        tile_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call golden reference implementation."""
        golden = FusedInferAttentionScoreGolden(
            num_heads_q=self._num_heads_q,
            num_heads_kv=self._num_heads_kv,
            head_dim=self._head_dim,
            softmax_scale=self._softmax_scale,
            tile_size=tile_size,
        )
        return golden.forward_golden(
            q, k, v,
            query_rope=query_rope,
            key_rope=key_rope,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            softmax_scale=softmax_scale,
            sparse_mode=sparse_mode if sparse_mode is not None else self._sparse_mode,
            atten_mask=atten_mask,
            layout_q=layout_q if layout_q is not None else self._layout_q,
            layout_kv=layout_kv if layout_kv is not None else self._layout_kv,
            layout_out=layout_out if layout_out is not None else self._layout_out,
            return_softmax_lse=return_softmax_lse,
            device_mode=device_mode,
            compute_mode=compute_mode,
        )
