"""CPU Golden and an independent Torch Oracle for KPool.

The first implementation intentionally rejects non-empty RoPE inputs because the
first KPool kernel version keeps the interface but does not implement RoPE.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


SUPPORTED_RATIOS = (2, 4, 8, 16, 32, 64, 128)
DEFAULT_NORM_EPS = 1e-6
HEAD_DIM = 128
HIDDEN_SIZE = 4096
ROPE_DIM = 64


def _as_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _split_hidden(
    hidden_states: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
) -> Tuple[list[torch.Tensor], list[int]]:
    if hidden_states.dim() == 3:
        if seq_lens is not None:
            raise ValueError("seq_lens must be absent for BSH")
        return [hidden_states[index] for index in range(hidden_states.size(0))], [
            int(hidden_states.size(1))
        ] * hidden_states.size(0)
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states rank must be 2 (TH) or 3 (BSH)")
    if seq_lens is None:
        raise ValueError("seq_lens is required for TH")
    lengths = _as_list(seq_lens)
    if sum(lengths) != hidden_states.size(0):
        raise ValueError("sum(seq_lens) must equal hidden_states.shape[0]")
    chunks = []
    offset = 0
    for length in lengths:
        chunks.append(hidden_states[offset : offset + length])
        offset += length
    return chunks, lengths


def validate_inputs(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> None:
    if cmp_ratio not in SUPPORTED_RATIOS:
        raise ValueError(f"unsupported cmp_ratio={cmp_ratio}")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    if rotary_mode not in (0, 1):
        raise ValueError("rotary_mode must be 0 or 1")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must be BF16")
    if wk.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        raise TypeError("wk and gate_weight must be BF16")
    if ape.dtype != torch.float32:
        raise TypeError("ape must be FP32")
    if state_cache.dtype != torch.float32:
        raise TypeError("state_cache must be FP32")
    if cache_block_table.dtype != torch.int32 or start_pos.dtype != torch.int32:
        raise TypeError("cache_block_table and start_pos must be INT32")
    if hidden_states.dim() not in (2, 3):
        raise ValueError("hidden_states rank must be 2 or 3")
    if wk.dim() != 2 or gate_weight.shape != wk.shape:
        raise ValueError("wk and gate_weight must have the same rank-2 shape")
    d, h = wk.shape
    if hidden_states.size(-1) != h:
        raise ValueError("hidden_states last dimension must match wk.shape[1]")
    if ape.shape != (cmp_ratio, d):
        raise ValueError("ape shape must be [cmp_ratio, D]")
    if state_cache.dim() != 3 or state_cache.size(-1) != 2 * d:
        raise ValueError("state_cache shape must be [block_num, block_size, 2*D]")
    if cache_block_table.dim() != 2:
        raise ValueError("cache_block_table shape must be [B, max_block_num_per_batch]")
    if start_pos.dim() != 1 or start_pos.numel() != cache_block_table.size(0):
        raise ValueError("start_pos shape must be [B]")
    if enforce_model_dims and (h, d) != (HIDDEN_SIZE, HEAD_DIM):
        raise ValueError("first version requires H=4096 and D=128")
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be passed as a pair")
    if norm_weight is not None:
        if norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32:
            raise TypeError("norm parameters must be FP32")
        if norm_weight.shape != (d,) or norm_bias.shape != (d,):
            raise ValueError("norm parameters must have shape [D]")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must be passed as a pair")
    if cos is not None:
        if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
            raise TypeError("cos and sin must be BF16")
        if cos.shape != sin.shape:
            raise ValueError("cos and sin must have the same shape")
        raise NotImplementedError("KPool RoPE is reserved but not implemented in v1")
    if seq_lens is not None and seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be INT32")
    batch_hidden, lengths = _split_hidden(hidden_states, seq_lens)
    if cache_block_table.size(0) != len(batch_hidden):
        raise ValueError("cache_block_table batch does not match hidden_states")
    if any(int(value) < 0 for value in _as_list(start_pos)):
        raise ValueError("start_pos must be non-negative")
    if any(int(value) < 0 for value in _as_list(cache_block_table)):
        raise ValueError("cache_block_table must be non-negative")
    block_num = state_cache.size(0)
    if block_num == 0 or state_cache.size(1) == 0:
        raise ValueError("state_cache must not be empty")
    if any(int(value) >= block_num for value in _as_list(cache_block_table)):
        raise ValueError("cache_block_table contains an out-of-range physical block")
    capacity = cache_block_table.size(1) * state_cache.size(1)
    for batch, length in enumerate(lengths):
        if int(start_pos[batch]) + length > capacity:
            raise ValueError("updated sequence length exceeds logical cache capacity")
        if int(start_pos[batch]) + length > 0:
            required_blocks = (int(start_pos[batch]) + length + state_cache.size(1) - 1) // state_cache.size(1)
            if any(int(value) == 0 for value in cache_block_table[batch, :required_blocks]):
                raise ValueError("physical block 0 is invalid for a used logical block")


def _project(
    hidden: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    norm_bias: Optional[torch.Tensor],
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_fp32 = hidden.float()
    k = torch.matmul(hidden_fp32, wk.float().transpose(-1, -2)).to(hidden.dtype)
    gate = torch.matmul(hidden_fp32, gate_weight.float().transpose(-1, -2)).to(hidden.dtype)
    if norm_weight is not None:
        k_fp32 = k.float()
        mean = k_fp32.mean(dim=-1, keepdim=True)
        variance = (k_fp32 - mean).square().mean(dim=-1, keepdim=True)
        k = (((k_fp32 - mean) / torch.sqrt(variance + norm_eps)) *
             norm_weight + norm_bias).to(hidden.dtype)
    return k, gate


def _write_cache(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    batch: int,
    k: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    block_size = state_cache.size(1)
    for token in range(k.size(0)):
        logical_pos = int(start_pos[batch]) + token
        logical_block, offset = divmod(logical_pos, block_size)
        physical_block = int(cache_block_table[batch, logical_block])
        if physical_block == 0:
            raise ValueError("attempted to write through invalid physical block 0")
        state_cache[physical_block, offset, : k.size(-1)] = k[token].float()
        state_cache[physical_block, offset, k.size(-1) :] = gate[token].float()


def _read_cache(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    batch: int,
    logical_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_size = state_cache.size(1)
    logical_block, offset = divmod(logical_pos, block_size)
    physical_block = int(cache_block_table[batch, logical_block])
    if physical_block == 0:
        raise ValueError("attempted to read through invalid physical block 0")
    width = state_cache.size(-1) // 2
    return state_cache[physical_block, offset, :width], state_cache[physical_block, offset, width:]


def _pool_one_batch_loop(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    batch: int,
    updated_len: int,
    cmp_ratio: int,
    ape: torch.Tensor,
    output: torch.Tensor,
) -> None:
    width = state_cache.size(-1) // 2
    valid_pool_count = updated_len // cmp_ratio
    for pool in range(valid_pool_count):
        keys = []
        gates = []
        for relative in range(cmp_ratio):
            key, gate = _read_cache(
                state_cache, cache_block_table, batch, pool * cmp_ratio + relative
            )
            keys.append(key)
            gates.append(gate + ape[relative])
        key_tensor = torch.stack(keys, dim=0)
        logits = torch.stack(gates, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        probabilities = probabilities.to(torch.bfloat16).float()
        output[batch, pool] = (probabilities * key_tensor).sum(dim=0).to(torch.bfloat16)


def run_k_pool_golden(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reference implementation using scalar logical Cache traversal."""
    validate_inputs(
        hidden_states, wk, gate_weight, ape, state_cache, cache_block_table, start_pos,
        norm_weight, norm_bias, cos, sin, seq_lens, cmp_ratio, norm_eps, rotary_mode,
        enforce_model_dims,
    )
    batch_hidden, lengths = _split_hidden(hidden_states, seq_lens)
    k_parts = []
    gate_parts = []
    for hidden in batch_hidden:
        k, gate = _project(hidden, wk, gate_weight, norm_weight, norm_bias, norm_eps)
        k_parts.append(k)
        gate_parts.append(gate)
    for batch, (k, gate) in enumerate(zip(k_parts, gate_parts)):
        _write_cache(state_cache, cache_block_table, start_pos, batch, k, gate)

    pcap = (cache_block_table.size(1) * state_cache.size(1) + cmp_ratio - 1) // cmp_ratio
    output = torch.zeros(
        (len(batch_hidden), pcap, wk.size(0)), dtype=torch.bfloat16, device=state_cache.device
    )
    for batch, length in enumerate(lengths):
        updated_len = int(start_pos[batch]) + length
        _pool_one_batch_loop(
            state_cache, cache_block_table, batch, updated_len, cmp_ratio, ape, output
        )
    intermediates = {
        "k_projection": torch.cat(k_parts, dim=0) if k_parts else torch.empty(0),
        "gate_projection": torch.cat(gate_parts, dim=0) if gate_parts else torch.empty(0),
        "state_cache": state_cache,
    }
    return output, intermediates


def run_k_pool_oracle(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent vectorized Pool read/softmax implementation."""
    validate_inputs(
        hidden_states, wk, gate_weight, ape, state_cache, cache_block_table, start_pos,
        norm_weight, norm_bias, cos, sin, seq_lens, cmp_ratio, norm_eps, rotary_mode,
        enforce_model_dims,
    )
    batch_hidden, lengths = _split_hidden(hidden_states, seq_lens)
    for batch, hidden in enumerate(batch_hidden):
        k, gate = _project(hidden, wk, gate_weight, norm_weight, norm_bias, norm_eps)
        _write_cache(state_cache, cache_block_table, start_pos, batch, k, gate)

    batch_size = len(batch_hidden)
    block_size = state_cache.size(1)
    pcap = (cache_block_table.size(1) * block_size + cmp_ratio - 1) // cmp_ratio
    logical_positions = torch.arange(pcap * cmp_ratio, device=state_cache.device)
    logical_capacity = cache_block_table.size(1) * block_size
    safe_positions = logical_positions.clamp(max=logical_capacity - 1)
    block_ids = safe_positions // block_size
    offsets = safe_positions % block_size
    physical = cache_block_table[:, block_ids]
    gathered = state_cache[physical, offsets]
    gathered = gathered.view(batch_size, pcap, cmp_ratio, -1)
    keys = gathered[..., : wk.size(0)]
    gates = gathered[..., wk.size(0) :]
    updated_lens = start_pos.to(torch.long) + torch.tensor(lengths, device=start_pos.device)
    valid = logical_positions.view(1, pcap, cmp_ratio) < updated_lens[:, None, None]
    logits = gates + ape.view(1, 1, cmp_ratio, -1)
    logits = logits.masked_fill(~valid[..., None], float("-inf"))
    probabilities = torch.nan_to_num(torch.softmax(logits, dim=2)).to(torch.bfloat16).float()
    pooled = (probabilities * keys).sum(dim=2).to(torch.bfloat16)
    complete = valid.all(dim=2)
    pooled = pooled.masked_fill(~complete[..., None], 0)
    return pooled, state_cache


def compare_golden_and_oracle(
    golden_output: torch.Tensor,
    oracle_output: torch.Tensor,
    golden_cache: torch.Tensor,
    oracle_cache: torch.Tensor,
) -> None:
    torch.testing.assert_close(golden_output, oracle_output, rtol=0, atol=0)
    torch.testing.assert_close(golden_cache, oracle_cache, rtol=0, atol=0)
