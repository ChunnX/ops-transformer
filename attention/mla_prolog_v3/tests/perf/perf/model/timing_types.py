#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result dataclasses for matmul and vector operation timing."""

from dataclasses import dataclass


@dataclass
class MatMulTiming:
    name: str
    M: int
    K: int
    N: int
    active_cores: int
    load_A_hbm_us: float = 0.0
    load_A_l2_us: float = 0.0
    load_B_hbm_us: float = 0.0
    cube_us: float = 0.0
    fixpipe_us: float = 0.0
    total_us: float = 0.0
    bound: str = ""
    a_source: str = "HBM"
    # Inner-loop detail fields (populated by estimate_matmul_detailed)
    l0a_us: float = 0.0
    l0b_us: float = 0.0
    mmad_us: float = 0.0
    mte1_outer_us: float = 0.0
    kL1_loops: int = 0
    inner_bound: str = ""

    @property
    def mte1_us(self) -> float:
        return max(self.load_A_hbm_us if self.a_source == "HBM" else self.load_A_l2_us,
                   self.load_B_hbm_us)


@dataclass
class VectorOpTiming:
    name: str
    mte2_load_us: float = 0.0
    vec_compute_us: float = 0.0
    mte3_store_us: float = 0.0
    total_us: float = 0.0
    bound: str = ""
