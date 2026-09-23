#!/usr/bin/env python3
"""Compare Kimi K3 MLA decode precision: merged flash path vs 5e457e3 baseline.

The two decode pipelines under comparison:

Path A (current, ``master_dspark_flash_attn_v3``): ``mla_prolog`` writes one
merged 576-wide (ckv 512 + kr 64) PA_NZ latent cache, then
``flash_mla_with_kvcache`` attends over it with a 576-wide absorbed query.

Path B (verified baseline, commit ``5e457e3``): ``mla_prolog`` writes separate
nope/rope caches, then ``npu_fused_infer_attention_score_v2`` (``TND_NTD``
absorb) attends over them.

Both paths consume the same weights and per-step inputs.  The initial cache is
pre-filled with identical random latents, then each decode step appends tokens
through its own prolog.  Per step the script reports max-abs / max-rel /
mean-abs / cosine diffs for the prolog queries, the just-written cache slots,
and the final attention output, so a divergence can be attributed to the prolog
cache write or to the attention kernel.

Cache layout note: the BF16 PA caches are allocated with a logical 4D ND shape
but their backing storage follows the physical NZ tile order
(``[block, dim/16, block_size, 16]``) that the prolog writes and the attention
kernels read, so the prefill writes and the comparison reads below go through
the same tile scatter/gather as ``KimiMLAAttention._write_latent_cache``.

Defaults mirror the target inference scenario (100K context, DSpark verify
width ``q_len=7+1``, npugraph_ex capture).  Run in a CANN/PyTorch-NPU
environment::

    python models/kimi_k3/tools/mla_decode_precision_compare.py --device 0
    python models/kimi_k3/tools/mla_decode_precision_compare.py --no-use-graph
    python models/kimi_k3/tools/mla_decode_precision_compare.py --q-len 1
    python models/kimi_k3/tools/mla_decode_precision_compare.py --prolog-api v3
"""

import argparse
import json
from pathlib import Path
from typing import Any

# Imported at module scope: the path classes subclass torch.nn.Module at
# import time, and this script only ever runs inside a CANN/PyTorch-NPU env.
import torch
import torch_npu
import cann_ops_transformer  # noqa: F401  # registers torch ops


KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
NUM_HEADS = 96
RMS_NORM_EPS = 1.0e-5
BLOCK_SIZE = 128
NZ_DIM = 16
PROLOG_WEIGHT_SCALE = 0.02


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1,
                        help="MLA decode rows (mla_batch_per_rank).")
    parser.add_argument("--q-len", type=int, default=8,
                        help="Per-request query width; 8 models DSpark "
                             "verify (next_n=7), 1 models plain decode.")
    parser.add_argument("--seq-len", type=int, default=102400,
                        help="Pre-filled cache length before the compared "
                             "decode steps (100K target scenario).")
    parser.add_argument("--steps", type=int, default=4,
                        help="Decode steps to compare; each step appends "
                             "q-len tokens per request.")
    parser.add_argument("--accept-per-step", type=int, default=None,
                        help="Committed tokens per step; defaults to q-len. "
                             "A value below q-len mirrors DSpark verify: the "
                             "next round rewrites the uncommitted slots.")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--prolog-api", choices=("v3", "cann", "both"),
                        default="both",
                        help="Prolog interface used by path A; each variant "
                             "gets its own cache instance. Note: npu_mla_"
                             "prolog_v3 with ckvkr_repo_mode=1 requires a "
                             "torch_npu whose wrapper allocates the merged "
                             "576-wide query; 2.9.0.post7 fails tiling with "
                             "a query shape error (wrapper allocates 512).")
    parser.add_argument("--use-graph", dest="use_graph",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Compile each path with the npugraph_ex backend "
                             "(matches the production decode capture).")
    parser.add_argument("--static-kernel", dest="static_kernel",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Pass static_kernel_compile=True to the "
                             "npugraph_ex options, matching infer.yaml's "
                             "enable_static_kernel.")
    parser.add_argument("--flash-max-kv", choices=("neg1", "bound"),
                        default="neg1",
                        help="max_seqlen_kv passed to the flash metadata: "
                             "'neg1' mirrors the model, 'bound' passes the "
                             "constant cache upper bound (diagnoses whether "
                             "the static-kernel metadata crash is caused by "
                             "the -1 sentinel).")
    parser.add_argument("--cache-compile", dest="cache_compile",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Wrap with npugraph_ex.inference.cache_compile "
                             "like the production runner (implies graph "
                             "mode).")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("./compile_cache/mla_compare"),
                        help="Cache directory for --cache-compile.")
    parser.add_argument("--weights-file", type=Path, default=None,
                        help="Optional checkpoint safetensors shard: load "
                             "the first MLA layer's real weights instead of "
                             "random ones to stress the kernels with real "
                             "weight distributions.")
    parser.add_argument(
        "--num-heads", type=int, default=None,
        help="Override attention heads (use the local head count for a TP shard)."
    )
    parser.add_argument("--config-json", type=Path, default=None,
                        help="Optional checkpoint config.json whose "
                             "text_config overrides the built-in dims.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-dir", type=Path, default=None,
                        help="Optionally save per-path queries, caches and "
                             "attention outputs as CPU .pt files.")
    return parser.parse_args()


def _load_dims(args: argparse.Namespace) -> dict[str, int]:
    dims = {
        "hidden_size": HIDDEN_SIZE,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "num_attention_heads": NUM_HEADS,
        "rms_norm_eps": RMS_NORM_EPS,
    }
    if args.config_json is not None:
        config = json.loads(args.config_json.read_text())
        text_config = config.get("text_config", config)
        for key in dims:
            if key in text_config:
                dims[key] = text_config[key]
    if args.num_heads is not None:
        dims["num_attention_heads"] = args.num_heads
    return dims


def _check_args(args: argparse.Namespace, dims: dict[str, int]) -> None:
    if args.batch <= 0 or args.q_len <= 0 or args.steps <= 0:
        raise ValueError("batch, q-len and steps must be positive")
    if dims["num_attention_heads"] <= 0:
        raise ValueError("num-heads must be positive")
    if args.seq_len < 0:
        raise ValueError("seq-len must be non-negative")
    if args.block_size % NZ_DIM or args.block_size < NZ_DIM:
        raise ValueError("block-size must be a multiple of 16")
    if dims["kv_lora_rank"] % NZ_DIM or dims["qk_rope_head_dim"] % NZ_DIM:
        raise ValueError("latent and rope widths must be multiples of 16")
    if args.q_len > min(args.block_size, 2048):
        raise ValueError("q-len must fit inside one block and the 2048 mask")
    if args.accept_per_step is not None and not (
        0 < args.accept_per_step <= args.q_len
    ):
        raise ValueError(
            "accept-per-step must be in (0, q-len]; a smaller value mirrors "
            "DSpark verify's partial commits"
        )


def _as_nz(torch_npu: Any, tensor: Any) -> Any:
    # 29 is ACL_FORMAT_FRACTAL_NZ, required for the prolog matrix weights.
    return torch_npu.npu_format_cast(tensor.contiguous(), 29)


def _build_weights(torch: Any, torch_npu: Any, dims: dict[str, int],
                   device: str) -> dict[str, Any]:
    """Random BF16 weights shared by both paths, in the layouts they expect."""
    heads = dims["num_attention_heads"]
    q_head_dim = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
    kv_b_head_dim = dims["qk_nope_head_dim"] + dims["v_head_dim"]
    kv_dim = dims["kv_lora_rank"]

    def proj(*shape: int) -> Any:
        return (torch.randn(*shape, dtype=torch.bfloat16, device=device)
                * PROLOG_WEIGHT_SCALE)

    # kv_b_proj and q_b_proj have different per-head widths.  Production
    # _split_kv_b_proj uses [qk_nope, v_head] for kv_b; qk_rope belongs only
    # to q_b_proj.  Keeping these contracts separate is essential for a
    # valid post-Flash V projection comparison.
    kv_b_raw = proj(heads * kv_b_head_dim, kv_dim)
    weight = kv_b_raw.t().contiguous().view(kv_dim, heads, kv_b_head_dim)
    w_k, w_v = weight.split(
        [dims["qk_nope_head_dim"], dims["v_head_dim"]], dim=-1
    )

    # Keep an ND copy of the kv_a projection for the prefill-style reference
    # writer: it mirrors what Prefill's kv_a_proj_with_mqa consumes, while the
    # prologs read the NZ cast of a clone of the same values.
    dkv_nd = proj(dims["hidden_size"], kv_dim + dims["qk_rope_head_dim"])

    return {
        "weight_dq": _as_nz(
            torch_npu, proj(dims["hidden_size"], dims["q_lora_rank"])
        ),
        "weight_uq_qr": _as_nz(
            torch_npu, proj(dims["q_lora_rank"], heads * q_head_dim)
        ),
        # Absorbed K stays ND: [heads, qk_nope, kv_lora].
        "weight_uk": w_k.permute(1, 2, 0).contiguous(),
        "weight_dkv_kr": _as_nz(torch_npu, dkv_nd.clone()),
        # ND twin of weight_dkv_kr for the prefill-style reference writer.
        "weight_dkv_nd": dkv_nd,
        # Absorbed V stays ND: [heads, kv_lora, v_head].
        "weight_v": w_v.transpose(0, 1).contiguous(),
        "gamma_cq": torch.ones(
            dims["q_lora_rank"], dtype=torch.bfloat16, device=device
        ),
        "gamma_ckv": torch.ones(
            kv_dim, dtype=torch.bfloat16, device=device
        ),
    }


def _load_real_weights(torch: Any, torch_npu: Any, args: argparse.Namespace,
                       dims: dict[str, int], device: str) -> dict[str, Any]:
    """Load the first MLA layer's real weights from a checkpoint shard.

    Key names follow the checkpoint (nn.Linear [out, in]); the mapped layouts
    match what the two prolog/attention paths consume, the same way
    ``KimiLinearForCausalLM.load_weights`` / ``_split_kv_b_proj`` map them.
    """
    from safetensors.torch import load_file

    tensors = load_file(str(args.weights_file))
    prefix = None
    for name in tensors:
        if name.endswith("self_attn.q_a_proj.weight"):
            prefix = name[: -len("q_a_proj.weight")]
            break
    if prefix is None:
        raise SystemExit(
            f"no MLA layer (self_attn.q_a_proj.weight) found in "
            f"{args.weights_file}; pick a shard that contains one"
        )
    print(f"loading real MLA weights from {prefix}*")

    heads = dims["num_attention_heads"]
    q_head_dim = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
    kv_b_head_dim = dims["qk_nope_head_dim"] + dims["v_head_dim"]
    kv_dim = dims["kv_lora_rank"]

    def get(suffix: str) -> Any:
        tensor = tensors[prefix + suffix]
        return tensor.to(device=device, dtype=torch.bfloat16)

    kv_b_raw = get("kv_b_proj.weight")  # [heads*(qk_nope+v), kv_dim]
    expected_shape = (heads * kv_b_head_dim, kv_dim)
    if tuple(kv_b_raw.shape) != expected_shape:
        raise ValueError(
            "kv_b_proj.weight shape mismatch: "
            f"got {tuple(kv_b_raw.shape)}, expected {expected_shape}"
        )
    weight = kv_b_raw.t().contiguous().view(kv_dim, heads, kv_b_head_dim)
    w_k, w_v = weight.split(
        [dims["qk_nope_head_dim"], dims["v_head_dim"]], dim=-1
    )

    dkv_nd = get("kv_a_proj_with_mqa.weight").t().contiguous()
    return {
        "weight_dq": _as_nz(torch_npu, get("q_a_proj.weight").t().contiguous()),
        "weight_uq_qr": _as_nz(torch_npu, get("q_b_proj.weight").t().contiguous()),
        "weight_uk": w_k.permute(1, 2, 0).contiguous(),
        "weight_dkv_kr": _as_nz(torch_npu, dkv_nd.clone()),
        "weight_dkv_nd": dkv_nd,
        "weight_v": w_v.transpose(0, 1).contiguous(),
        "gamma_cq": get("q_a_layernorm.weight").contiguous(),
        "gamma_ckv": get("kv_a_layernorm.weight").contiguous(),
    }


class StepPlan:
    """Paged-cache geometry and per-step metadata shared by both paths."""

    def __init__(self, torch_module: Any, args: argparse.Namespace,
                 device: str) -> None:
        self.torch = torch_module
        self.args = args
        self.device = device
        self.total_len = args.seq_len + args.steps * args.q_len
        self.blocks_per_seq = -(-self.total_len // args.block_size)
        self.block_num = 1 + args.batch * self.blocks_per_seq

        # Block 0 stays null, mirroring CacheData/AttnMetaData.
        self.block_table = torch_module.arange(
            1,
            1 + args.batch * self.blocks_per_seq,
            dtype=torch.int32,
            device=device,
        ).view(args.batch, self.blocks_per_seq)

        # FIA bool mask (inverse lower-triangle) for sparse_mode=3 and the
        # flash int8 strict lower-triangle for mask_mode=3, matching
        # AttnMetaData.attention_mask / flash_attention_mask.
        self.attention_mask = ~torch_module.tril(
            torch_module.ones((2048, 2048), dtype=torch.bool, device=device)
        )
        self.flash_attention_mask = torch_module.tril(
            torch_module.ones((2048, 2048), dtype=torch.bool, device=device),
            diagonal=-1,
        ).to(torch.int8)

    def step_inputs(self, step: int) -> dict[str, Any]:
        """Slots, positions and kernel lengths for one decode step."""
        torch = self.torch
        args = self.args
        # DSpark verify commits fewer tokens than it writes: the written
        # window is always q_len wide, but kv advances by the accepted count,
        # so the next round rewrites the uncommitted slots.
        committed = args.accept_per_step or args.q_len
        kv_before = args.seq_len + step * committed
        positions = (
            torch.arange(args.q_len, dtype=torch.int64, device=self.device)
            .view(1, -1)
            + kv_before
        ).expand(args.batch, -1)
        block_ids = self.block_table.to(torch.int64).gather(
            1, torch.div(positions, args.block_size, rounding_mode="floor")
        )
        offsets = positions % args.block_size
        slots = (block_ids * args.block_size + offsets).view(-1)
        return {
            "slots": slots,
            "kv_before": kv_before,
            # flash_mla_with_kvcache: tensors.
            "cache_seqlens": torch.full(
                (args.batch,), kv_before + args.q_len,
                dtype=torch.int32, device=self.device,
            ),
            "cu_seqlens_q": torch.arange(
                0, (args.batch + 1) * args.q_len, args.q_len,
                dtype=torch.int32, device=self.device,
            ),
            # npu_fused_infer_attention_score_v2: Python lists.
            "cu_q_list": [args.q_len * (idx + 1) for idx in range(args.batch)],
            "kv_list": [kv_before + args.q_len] * args.batch,
        }


def _allocate_caches(torch: Any, args: argparse.Namespace, plan: StepPlan,
                     dims: dict[str, int], device: str) -> dict[str, Any]:
    """Allocate one zeroed cache set (both layouts)."""
    shape_blocks = (plan.block_num, args.block_size)
    return {
        # Path B (5e457e3): separate logical ND caches.
        "nope_cache": torch.zeros(
            (*shape_blocks, 1, dims["kv_lora_rank"]),
            dtype=torch.bfloat16, device=device,
        ),
        "rope_cache": torch.zeros(
            (*shape_blocks, 1, dims["qk_rope_head_dim"]),
            dtype=torch.bfloat16, device=device,
        ),
        # Path A: one merged logical PA_NZ cache plus the empty kr placeholder.
        "kv_cache": torch.zeros(
            (*shape_blocks, 1, dims["kv_lora_rank"] + dims["qk_rope_head_dim"]),
            dtype=torch.bfloat16, device=device,
        ),
        "kr_cache": torch.empty(
            (0, args.block_size, 1, dims["qk_rope_head_dim"]),
            dtype=torch.bfloat16, device=device,
        ),
    }


def _nz_rows(torch: Any, slots: Any, tile_count: int, block_size: int) -> Any:
    """Flat NZ row indices of every 16-lane tile for the given slots.

    Mirrors the tile index formula of ``KimiMLAAttention._write_latent_cache``:
    ``block_id * tile_count * block_size + tile * block_size + offset``.
    """
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots % block_size
    return (
        blocks[:, None] * (tile_count * block_size)
        + torch.arange(tile_count, device=slots.device)[None, :] * block_size
        + offsets[:, None]
    )


def _scatter_rows(torch: Any, cache: Any, slots: Any, values: Any,
                  block_size: int, dim: int) -> None:
    """Write logical [N, dim] rows into a PA_NZ cache's tile storage."""
    rows = _nz_rows(torch, slots, dim // NZ_DIM, block_size)
    cache.view(-1, NZ_DIM).index_copy_(
        0, rows.reshape(-1), values.reshape(-1, NZ_DIM)
    )


def _gather_rows(torch: Any, cache: Any, slots: Any, block_size: int,
                 dim: int) -> Any:
    """Read logical [N, dim] rows back from a PA_NZ cache's tile storage."""
    rows = _nz_rows(torch, slots, dim // NZ_DIM, block_size)
    gathered = cache.view(-1, NZ_DIM).index_select(0, rows.reshape(-1))
    return gathered.view(-1, dim // NZ_DIM, NZ_DIM).reshape(-1, dim)


def _prefill_caches(torch: Any, cache_sets: dict[str, dict[str, Any]],
                    plan: StepPlan, dims: dict[str, int]) -> None:
    """Fill positions [0, seq_len) of every cache set with identical latents."""
    args = plan.args
    kv_dim = dims["kv_lora_rank"]
    rope_dim = dims["qk_rope_head_dim"]
    for row in range(args.batch):
        latents = torch.randn(
            args.seq_len, kv_dim + rope_dim, dtype=torch.bfloat16,
            device=plan.device,
        )
        positions = torch.arange(
            args.seq_len, dtype=torch.int64, device=plan.device
        )
        # aclnnGather requires index dims to match the input dims, so the
        # block indices keep the block_table's 2D row shape here.
        block_ids = plan.block_table[row : row + 1].to(torch.int64).gather(
            1,
            torch.div(
                positions, args.block_size, rounding_mode="floor"
            ).view(1, -1),
        ).view(-1)
        slots = block_ids * args.block_size + positions % args.block_size
        for caches in cache_sets.values():
            _scatter_rows(
                torch, caches["nope_cache"],
                slots, latents[:, :kv_dim].contiguous(),
                args.block_size, kv_dim,
            )
            _scatter_rows(
                torch, caches["rope_cache"],
                slots, latents[:, kv_dim:].contiguous(),
                args.block_size, rope_dim,
            )
            _scatter_rows(
                torch, caches["kv_cache"], slots, latents,
                args.block_size, kv_dim + rope_dim,
            )


def _reference_write_rows(torch_npu: Any, hidden: Any, weights: dict[str, Any],
                          dims: dict[str, int]) -> Any:
    """Rows the Prefill writer would produce, mirroring
    ``KimiMLAAttention._write_latent_cache``: kv_a projection, ckv RMSNorm,
    then the merged row that lands in the paged cache.

    The real model's initial cache comes from this writer while decode appends
    through the prolog's internal write, so per-step slots are compared
    against these rows to expose any seam between the two writers.
    """
    kv_dim = dims["kv_lora_rank"]
    rope_dim = dims["qk_rope_head_dim"]
    compressed = hidden.to(torch.bfloat16) @ weights["weight_dkv_nd"]
    k_nope = torch_npu.npu_rms_norm(
        compressed[:, :kv_dim].contiguous(),
        weights["gamma_ckv"],
        dims["rms_norm_eps"],
    )[0]
    return torch.cat((k_nope, compressed[:, kv_dim:]), dim=-1).contiguous()


def _split_absorbed_query(q: Any, q_rope: Any, kv_dim: int,
                          rope_dim: int) -> tuple[Any, Any]:
    """Normalize prolog query outputs to separate (q_nope, q_rope)."""
    if q.shape[-1] == kv_dim + rope_dim:
        # Merged 576-wide query (cann prolog with ckvkr_repo_mode=1).
        return q[..., :kv_dim].contiguous(), q[..., kv_dim:].contiguous()
    if q_rope is None or q_rope.numel() == 0:
        return q.contiguous(), q.new_zeros(q.shape[0], q.shape[1], rope_dim)
    return q.contiguous(), q_rope.contiguous()


class _BaselinePath(torch.nn.Module):  # type: ignore[name-defined]
    """Path B: the 5e457e3 decode implementation.

    ``mla_prolog`` (no rope tensors, no ``ckvkr_repo_mode``) writes the
    separate nope/rope caches, then ``npu_fused_infer_attention_score_v2``
    (TND_NTD) attends with the absorbed query and ``kv_b_proj_decode_w_v``
    restores the value space.
    """

    def __init__(self, cfg: dict[str, Any], caches: dict[str, Any],
                 plan: StepPlan) -> None:
        super().__init__()
        self.cfg = cfg
        # Weights are read-only in every path, so the modules share the same
        # tensors; clone() would fail on the NZ-format prolog weights.
        for name, tensor in cfg["weights"].items():
            self.register_buffer(name, tensor, persistent=False)
        self.register_buffer("nope_cache", caches["nope_cache"], persistent=False)
        self.register_buffer("rope_cache", caches["rope_cache"], persistent=False)
        self.register_buffer("block_table", plan.block_table, persistent=False)
        self.register_buffer(
            "attention_mask", plan.attention_mask, persistent=False
        )

    def forward(self, hidden_states, cache_index, cu_q_list, kv_list):
        cfg = self.cfg
        heads = cfg["heads"]
        kv_dim = cfg["kv_dim"]
        rope_dim = cfg["rope_dim"]
        multi_token = cfg["q_len"] > 1

        # torch.ops.cann_ops_transformer.mla_prolog @ 5e457e3 (_mla_prolog):
        # nine positional inputs, no rope tensors, no ckvkr_repo_mode.
        q_nope, q_rope, _, _, _ = torch.ops.cann_ops_transformer.mla_prolog(
            hidden_states.contiguous(),
            self.weight_dq,
            self.weight_uq_qr,
            self.weight_uk,
            self.weight_dkv_kr,
            self.gamma_cq,
            self.gamma_ckv,
            self.nope_cache,
            self.rope_cache,
            cache_index=cache_index.reshape(-1).to(torch.int64),
            rmsnorm_epsilon_cq=cfg["eps"],
            rmsnorm_epsilon_ckv=cfg["eps"],
            cache_mode="PA_NZ",
            query_norm_flag=True,
            weight_quant_mode=0,
        )

        # _nz_cache_inputs: the BF16 caches are read through their NZ view.
        block_num, seq = self.nope_cache.shape[0], self.nope_cache.shape[1]
        nope_nz = self.nope_cache.view(block_num, 1, kv_dim // NZ_DIM, seq, NZ_DIM)
        rope_nz = self.rope_cache.view(
            block_num, 1, rope_dim // NZ_DIM, seq, NZ_DIM
        )

        # _forward_decode_prolog @ 5e457e3.
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q_nope,
            nope_nz,
            nope_nz,
            query_rope=q_rope,
            key_rope=rope_nz,
            num_query_heads=heads,
            num_key_value_heads=1,
            softmax_scale=cfg["softmax_scale"],
            input_layout="TND_NTD",
            sparse_mode=3 if multi_token else 0,
            atten_mask=self.attention_mask if multi_token else None,
            actual_seq_qlen=cu_q_list,
            actual_seq_kvlen=kv_list,
            block_table=self.block_table,
            block_size=cfg["block_size"],
        )
        # Keep the raw FIA latent so the probe can distinguish an attention
        # mismatch from the following V projection/transpose contract.
        raw_output = output
        output = torch_npu.npu_transpose_batchmatmul(
            raw_output[:heads],
            self.weight_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        return (
            q_nope,
            q_rope,
            raw_output,
            output.reshape(-1, heads * cfg["v_head_dim"]),
        )


class _MergedFlashPath(torch.nn.Module):  # type: ignore[name-defined]
    """Path A: the current merged-cache decode implementation.

    ``mla_prolog`` (``ckvkr_repo_mode=1``) writes one merged 576-wide PA_NZ
    cache, then ``flash_mla_with_kvcache`` attends over its NZ view with the
    merged 576-wide query.
    """

    def __init__(self, cfg: dict[str, Any], caches: dict[str, Any],
                 plan: StepPlan, prolog_api: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prolog_api = prolog_api
        self.q_len = cfg["q_len"]
        # Shared read-only weights; see _BaselinePath.__init__.
        for name, tensor in cfg["weights"].items():
            self.register_buffer(name, tensor, persistent=False)
        self.register_buffer("kv_cache", caches["kv_cache"], persistent=False)
        self.register_buffer("kr_cache", caches["kr_cache"], persistent=False)
        self.register_buffer("block_table", plan.block_table, persistent=False)
        self.register_buffer(
            "flash_attention_mask", plan.flash_attention_mask, persistent=False
        )

    def forward(self, hidden_states, cache_index, cache_seqlens, cu_seqlens_q):
        cfg = self.cfg
        heads = cfg["heads"]
        kv_dim = cfg["kv_dim"]
        rope_dim = cfg["rope_dim"]
        merged_dim = kv_dim + rope_dim
        q_len = self.q_len

        if self.prolog_api == "v3":
            # Identity rope: K3's MLA is NoPE; V3 still requires the tensors
            # (mirrors KimiMLAAttention._mla_prolog on the current branch).
            rope_sin = hidden_states.new_zeros((hidden_states.shape[0], rope_dim))
            rope_cos = hidden_states.new_ones((hidden_states.shape[0], rope_dim))
            outputs = torch_npu.npu_mla_prolog_v3(
                token_x=hidden_states.contiguous(),
                weight_dq=self.weight_dq,
                weight_uq_qr=self.weight_uq_qr,
                weight_uk=self.weight_uk,
                weight_dkv_kr=self.weight_dkv_kr,
                rmsnorm_gamma_cq=self.gamma_cq,
                rmsnorm_gamma_ckv=self.gamma_ckv,
                rope_sin=rope_sin,
                rope_cos=rope_cos,
                kv_cache=self.kv_cache,
                kr_cache=self.kr_cache,
                cache_index=cache_index.reshape(-1).to(torch.int64),
                rmsnorm_epsilon_cq=cfg["eps"],
                rmsnorm_epsilon_ckv=cfg["eps"],
                cache_mode="PA_NZ",
                query_norm_flag=True,
                weight_quant_mode=0,
                ckvkr_repo_mode=1,
                kv_cache_quant_mode=0,
                query_quant_mode=0,
                quant_scale_repo_mode=0,
                qc_qr_scale=1.0,
                kc_scale=1.0,
            )
        else:
            # cann prolog with the merged repo mode (post-d9898bb contract).
            outputs = torch.ops.cann_ops_transformer.mla_prolog(
                hidden_states.contiguous(),
                self.weight_dq,
                self.weight_uq_qr,
                self.weight_uk,
                self.weight_dkv_kr,
                self.gamma_cq,
                self.gamma_ckv,
                self.kv_cache,
                self.kr_cache,
                cache_index=cache_index.reshape(-1).to(torch.int64),
                rmsnorm_epsilon_cq=cfg["eps"],
                rmsnorm_epsilon_ckv=cfg["eps"],
                cache_mode="PA_NZ",
                query_norm_flag=True,
                weight_quant_mode=0,
                ckvkr_repo_mode=1,
            )
        q_nope, q_rope = _split_absorbed_query(outputs[0], outputs[1],
                                               kv_dim, rope_dim)
        merged_q = torch.cat((q_nope, q_rope), dim=-1).contiguous()

        # _flash_mla_attention: the logical PA cache is read through its
        # physical PA_NZ view.
        block_num, seq = self.kv_cache.shape[0], self.kv_cache.shape[1]
        kv_cache_nz = self.kv_cache.view(
            block_num, 1, merged_dim // NZ_DIM, seq, NZ_DIM
        )
        mask_mode = 3 if q_len > 1 else 0
        # 'neg1' mirrors the model's sentinel; 'bound' passes a constant
        # cache upper bound instead (a per-step exact value would recompile
        # the graph every round).
        max_seqlen_kv = (
            cfg["flash_max_kv_bound"] if cfg.get("flash_max_kv_real") else -1
        )
        metadata = torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
            cache_seqlens,
            num_heads_q=heads,
            num_heads_kv=1,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            max_seqlen_q=q_len,
            max_seqlen_kv=max_seqlen_kv,
            head_dim_qk=merged_dim,
            head_dim_v=kv_dim,
            mask_mode=mask_mode,
            layout_q="TND",
        )
        output, _ = torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            merged_q,
            kv_cache_nz,
            block_table=self.block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            attn_mask=self.flash_attention_mask if mask_mode == 3 else None,
            metadata=metadata,
            head_dim_v=kv_dim,
            softmax_scale=cfg["softmax_scale"],
            mask_mode=mask_mode,
            max_seqlen_q=q_len,
            max_seqlen_kv=max_seqlen_kv,
            layout_q="TND",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=False,
        )
        # Keep the raw Flash MLA latent before device alignment and V
        # projection for a direct comparison with the legacy FIA output.
        raw_output = output
        output = output.to(device=self.weight_v.device)
        output = torch_npu.npu_transpose_batchmatmul(
            output,
            self.weight_v,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        return (
            q_nope,
            q_rope,
            raw_output,
            output.reshape(-1, heads * cfg["v_head_dim"]),
        )


def _latent_tnd(torch: Any, latent: Any, heads: int) -> Any:
    """Normalize FIA's [H,T,D] latent to Flash MLA's [T,H,D]."""
    if latent.dim() == 3 and latent.shape[0] == heads:
        return latent.permute(1, 0, 2).contiguous()
    return latent.contiguous()


def _diff_stats(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    a = actual.detach().float()
    b = expected.detach().float()
    diff = (a - b).abs()
    rel = diff / b.abs().clamp_min(1.0e-6)
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    return {
        "max_abs": diff.max().item(),
        "max_rel": rel.max().item(),
        "mean_abs": diff.mean().item(),
        # mean diff normalized by the output magnitude: severity of the
        # divergence independent of the random draw's scale.
        "norm_mean": (diff.mean() / b.abs().mean().clamp_min(1.0e-12)).item(),
        "cos": cos,
    }


def _print_row(label: str, stats: dict[str, float]) -> None:
    print(
        f"    {label:<11} max_abs={stats['max_abs']:.6e} "
        f"mean_abs={stats['mean_abs']:.6e} "
        f"norm_mean={stats['norm_mean']:.3e} cos={stats['cos']:.8f}"
    )


def _compile_path(torch: Any, path: torch.nn.Module,
                  static_kernel: bool = False, cache_compile: bool = False,
                  cache_dir: Any = None) -> torch.nn.Module:  # type: ignore[name-defined]
    # Mirrors KimiK3Runner.graph_compile's npugraph_ex options.
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    for buffer in path.buffers():
        if buffer.ndim > 0:
            torch._dynamo.mark_static(buffer)
    options = {
        "frozen_parameter": True,
        "static_kernel_compile": static_kernel,
    }
    if cache_compile:
        # Production wraps the decode entry with cache_compile instead of
        # plain torch.compile.
        return torch.npu.npugraph_ex.inference.cache_compile(
            path,
            cache_dir=str(cache_dir),
            dynamic=True,
            options=options,
        )
    return torch.compile(
        path,
        dynamic=True,
        fullgraph=True,
        backend="npugraph_ex",
        options=options,
    )


def main() -> None:
    args = _parse_args()
    dims = _load_dims(args)
    _check_args(args, dims)

    device = f"npu:{args.device}"
    torch.manual_seed(args.seed)
    torch_npu.npu.set_device(args.device)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
    print(
        f"batch={args.batch} q_len={args.q_len} seq_len={args.seq_len} "
        f"steps={args.steps} block_size={args.block_size} "
        f"heads={dims['num_attention_heads']} use_graph={args.use_graph} "
        f"static_kernel={args.static_kernel} "
        f"accept_per_step={args.accept_per_step or args.q_len} "
        f"flash_max_kv={args.flash_max_kv}"
    )

    q_head_dim = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
    cfg = {
        "weights": (
            _load_real_weights(torch, torch_npu, args, dims, device)
            if args.weights_file is not None
            else _build_weights(torch, torch_npu, dims, device)
        ),
        "softmax_scale": q_head_dim ** -0.5,
        "eps": dims["rms_norm_eps"],
        "heads": dims["num_attention_heads"],
        "kv_dim": dims["kv_lora_rank"],
        "rope_dim": dims["qk_rope_head_dim"],
        "v_head_dim": dims["v_head_dim"],
        "block_size": args.block_size,
        "q_len": args.q_len,
    }

    plan = StepPlan(torch, args, device)
    cfg["flash_max_kv_real"] = args.flash_max_kv == "bound"
    # Constant upper bound of the per-step cache lengths; a per-step exact
    # value would be a changing graph ATTR and recompile every round.
    cfg["flash_max_kv_bound"] = plan.total_len
    apis = ("v3", "cann") if args.prolog_api == "both" else (args.prolog_api,)

    # One independent cache set per compared pipeline: B, plus one per A
    # variant, so prolog writes never leak between pipelines.
    cache_sets = {"b": _allocate_caches(torch, args, plan, dims, device)}
    for api in apis:
        cache_sets[api] = _allocate_caches(torch, args, plan, dims, device)
    _prefill_caches(torch, cache_sets, plan, dims)

    baseline = _BaselinePath(cfg, cache_sets["b"], plan)
    merged_paths = {
        api: _MergedFlashPath(cfg, cache_sets[api], plan, api) for api in apis
    }
    if args.use_graph or args.cache_compile:
        baseline = _compile_path(
            torch, baseline, args.static_kernel,
            args.cache_compile, args.cache_dir,
        )
        merged_paths = {
            api: _compile_path(
                torch, path, args.static_kernel, args.cache_compile,
                args.cache_dir,
            )
            for api, path in merged_paths.items()
        }

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)

    kv_dim = dims["kv_lora_rank"]
    rope_dim = dims["qk_rope_head_dim"]
    block_size = args.block_size
    committed = args.accept_per_step or args.q_len
    for step in range(args.steps):
        inputs = plan.step_inputs(step)
        slots = inputs["slots"]
        hidden = torch.randn(
            args.batch * args.q_len, dims["hidden_size"],
            dtype=torch.bfloat16, device=device,
        )

        b_q_nope, b_q_rope, b_raw, b_out = baseline(
            hidden, slots, inputs["cu_q_list"], inputs["kv_list"]
        )
        torch.npu.synchronize()
        b_cache_nope = _gather_rows(
            torch, cache_sets["b"]["nope_cache"], slots, block_size, kv_dim
        ).float()
        b_cache_rope = _gather_rows(
            torch, cache_sets["b"]["rope_cache"], slots, block_size, rope_dim
        ).float()

        # What the Prefill writer (kv_a proj + RMSNorm) would have written
        # for this step's tokens; decode-written slots must match it closely
        # or the resident cache becomes inconsistent across the seam.
        ref_rows = _reference_write_rows(
            torch_npu, hidden, cfg["weights"], dims
        ).float()

        print(
            f"step {step} (written={args.q_len}, committed={committed}, "
            f"kv_after={inputs['kv_before'] + committed}):"
        )
        print("  B vs prefill-writer (decode write must match the initial "
              "cache writer):")
        _print_row(
            "cache_nope", _diff_stats(torch, b_cache_nope, ref_rows[:, :kv_dim])
        )
        _print_row(
            "cache_rope", _diff_stats(torch, b_cache_rope, ref_rows[:, kv_dim:])
        )
        for api, path in merged_paths.items():
            a_q_nope, a_q_rope, a_raw, a_out = path(
                hidden, slots, inputs["cache_seqlens"], inputs["cu_seqlens_q"]
            )
            torch.npu.synchronize()

            a_cache = _gather_rows(
                torch, cache_sets[api]["kv_cache"], slots, block_size,
                kv_dim + rope_dim,
            ).float()
            print(f"  A[{api}] vs B:")
            _print_row("q_nope", _diff_stats(torch, a_q_nope, b_q_nope))
            _print_row("q_rope", _diff_stats(torch, a_q_rope, b_q_rope))
            b_raw_tnd = _latent_tnd(torch, b_raw, cfg["heads"])
            a_raw_tnd = _latent_tnd(torch, a_raw, cfg["heads"])
            print(
                f"    raw_latent shapes: A={tuple(a_raw_tnd.shape)} "
                f"B={tuple(b_raw_tnd.shape)}"
            )
            if a_raw_tnd.shape == b_raw_tnd.shape:
                _print_row(
                    "raw_latent", _diff_stats(torch, a_raw_tnd, b_raw_tnd)
                )
            _print_row(
                "cache_nope",
                _diff_stats(torch, a_cache[:, :kv_dim], b_cache_nope),
            )
            _print_row(
                "cache_rope",
                _diff_stats(torch, a_cache[:, kv_dim:], b_cache_rope),
            )
            _print_row("attn_out", _diff_stats(torch, a_out, b_out))
            print(f"  A[{api}] vs prefill-writer:")
            _print_row(
                "cache_576", _diff_stats(torch, a_cache, ref_rows)
            )

            if args.dump_dir is not None:
                prefix = f"step{step}_{api}"
                torch.save(
                    a_q_nope.cpu(), args.dump_dir / f"{prefix}_q_nope.pt"
                )
                torch.save(
                    a_out.cpu(), args.dump_dir / f"{prefix}_attn_out.pt"
                )
                torch.save(
                    a_raw_tnd.cpu(), args.dump_dir / f"{prefix}_raw_latent.pt"
                )
        if args.dump_dir is not None:
            torch.save(b_q_nope.cpu(), args.dump_dir / f"step{step}_b_q_nope.pt")
            torch.save(b_out.cpu(), args.dump_dir / f"step{step}_b_attn_out.pt")
            torch.save(
                _latent_tnd(torch, b_raw, cfg["heads"]).cpu(),
                args.dump_dir / f"step{step}_b_raw_latent.pt",
            )

    if args.dump_dir is not None:
        torch.save(
            cache_sets["b"]["nope_cache"].cpu(),
            args.dump_dir / "b_nope_cache.pt",
        )
        for api in apis:
            torch.save(
                cache_sets[api]["kv_cache"].cpu(),
                args.dump_dir / f"a_{api}_kv_cache.pt",
            )
        print(f"saved tensors under {args.dump_dir}")

    print("MLA decode precision comparison finished")


if __name__ == "__main__":
    main()