#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OperatorParams: configuration of an MLA Prolog V3 invocation."""

from dataclasses import dataclass

from .quant import (
    QuantMode, resolve_quant_mode, dtype_size,
    is_full_quant, is_int8_quant, is_mxfp8,
)


@dataclass
class OperatorParams:
    """Parameters describing an MLA Prolog V3 invocation."""
    batch_size: int = 2
    seq_len: int = 4
    head_num: int = 8
    He: int = 7168
    Hcq: int = 1536
    Hckv: int = 512
    D: int = 128
    Dr: int = 64
    Nkv: int = 1
    block_size: int = 128
    weight_quant_mode: int = 0
    kv_cache_quant_mode: int = 0
    query_quant_mode: int = 0

    def __post_init__(self):
        """Validate field relationships at construction time.

        Defensive checks against obviously invalid configurations. Does not
        validate against the production-supported value sets (e.g. He must be
        in {1024, 2048, ...}) so the perf model can analyze hypothetical
        configurations beyond the kernel's actual support.
        """
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError(
                f"batch_size and seq_len must be positive, got "
                f"batch_size={self.batch_size}, seq_len={self.seq_len}")
        if self.head_num <= 0:
            raise ValueError(f"head_num must be positive, got {self.head_num}")
        if self.He <= 0 or self.Hcq <= 0 or self.Hckv <= 0:
            raise ValueError(
                f"He, Hcq, and Hckv must be positive, got "
                f"He={self.He}, Hcq={self.Hcq}, Hckv={self.Hckv}")
        if self.D <= 0 or self.Dr <= 0:
            raise ValueError(
                f"D and Dr must be positive, got D={self.D}, Dr={self.Dr}")
        if self.weight_quant_mode not in (0, 1, 2, 3):
            raise ValueError(
                f"weight_quant_mode must be in {{0,1,2,3}}, got {self.weight_quant_mode}")
        if self.kv_cache_quant_mode not in (0, 1, 2, 3):
            raise ValueError(
                f"kv_cache_quant_mode must be in {{0,1,2,3}}, got {self.kv_cache_quant_mode}")
        if self.query_quant_mode not in (0, 1):
            raise ValueError(
                f"query_quant_mode must be in {{0,1}}, got {self.query_quant_mode}")

    @property
    def T(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def N(self) -> int:
        return self.head_num

    @property
    def quant_mode(self) -> QuantMode:
        return resolve_quant_mode(self.weight_quant_mode, self.kv_cache_quant_mode)

    @property
    def weight_dtype_size(self) -> int:
        """Weight element size: 1 byte for INT8/FP8, 2 for BF16."""
        return dtype_size(self.quant_mode)

    @property
    def activation_dtype_size(self) -> int:
        """Activation (tokenX) element size."""
        if is_mxfp8(self.quant_mode):
            return 1
        return 2  # BF16

    @property
    def mm_output_dtype_size(self) -> int:
        """MatMul output size: 4 for INT32 (quant), 2 for BF16."""
        if is_full_quant(self.quant_mode) or is_mxfp8(self.quant_mode):
            return 4
        if is_int8_quant(self.quant_mode):
            return 4
        return 2
