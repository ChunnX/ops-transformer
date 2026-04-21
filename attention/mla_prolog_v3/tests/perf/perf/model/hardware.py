#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardware specifications for Ascend NPUs.

Defines:
- CacheSpec: per-core L1/L0/UB sizes
- AscendHWSpec: chip parameters (cube/vector throughput, bandwidth, cache, sync overhead)
- ASCEND_910B / ASCEND_950 instances and CHIP_REGISTRY lookup
"""

from dataclasses import dataclass, field


@dataclass
class CacheSpec:
    """Per-core on-chip cache sizes in bytes."""
    ub_size: int = 192 * 1024
    l1_size: int = 512 * 1024
    l0a_size: int = 64 * 1024
    l0b_size: int = 64 * 1024
    l0c_size: int = 128 * 1024


@dataclass
class AscendHWSpec:
    name: str = "Ascend910B"
    aic_num: int = 24
    aiv_num: int = 48

    # Chip-total performance (legacy 910B fields, kept for backward compat)
    cube_tflops_bf16: float = 320.0
    cube_tops_int8: float = 640.0
    cube_tflops_fp8: float = 640.0
    hbm_bw_bytes_per_sec: float = 1.2e12
    l2_bw_bytes_per_sec: float = 6.4e12
    fixpipe_bw_bytes_per_sec: float = 1.2e12
    vector_ops_per_sec_bf16: float = 10.0e12
    mte2_bw_bytes_per_sec: float = 1.2e12
    mte3_bw_bytes_per_sec: float = 1.2e12
    sync_overhead_us: float = 0.0
    scatter_efficiency: float = 0.4

    # Per-cycle specs (0.0 freq means "use chip-total mode")
    freq_ghz: float = 0.0
    cache: CacheSpec = field(default_factory=CacheSpec)

    # Per-core bandwidth in bytes/cycle
    hbm_bw_per_cycle: float = 0.0
    l2_bw_per_cycle: float = 0.0
    l1_to_l0a_per_cycle: float = 0.0
    l1_to_l0b_per_cycle: float = 0.0
    l0c_to_out_per_cycle: float = 0.0
    ub_to_l1_per_cycle: float = 0.0
    ub_to_reg_per_cycle: float = 0.0

    # Vector instruction throughput and IPC (0 = use legacy chip-total model)
    vec_instr_bytes_per_cycle: float = 0.0   # 256 for 950
    vec_ipc: float = 1.0                     # 1.3 for 950

    # MMAD throughput: multiply-accumulate ops per cycle per core
    mmad_ops_per_cycle_fp8: int = 0
    mmad_ops_per_cycle_fp16: int = 0

    # Chip total bandwidth caps (bytes/sec)
    chip_hbm_bw_bytes_per_sec: float = 0.0
    chip_l2_bw_bytes_per_sec: float = 0.0

    @property
    def has_cycle_spec(self) -> bool:
        return self.freq_ghz > 0.0

    @property
    def has_vec_instr_spec(self) -> bool:
        return self.vec_instr_bytes_per_cycle > 0.0

    @property
    def freq_hz(self) -> float:
        return self.freq_ghz * 1e9

    @property
    def cube_flops_bf16(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp16 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tflops_bf16 * 1e12

    @property
    def cube_ops_int8(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp16 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tops_int8 * 1e12

    @property
    def cube_flops_fp8(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp8 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tflops_fp8 * 1e12

    def per_core_bw(self, bus: str, active_cores: int) -> float:
        """Return effective per-core bandwidth in bytes/sec with contention.

        For HBM/L2: min(per_cycle * freq, chip_total / active_cores).
        For L1/L0 local buses: per_cycle * freq (no cross-core contention).
        """
        C = max(active_cores, 1)
        if bus == "hbm":
            raw = self.hbm_bw_per_cycle * self.freq_hz
            cap = self.chip_hbm_bw_bytes_per_sec / C
            return min(raw, cap)
        elif bus == "l2":
            raw = self.l2_bw_per_cycle * self.freq_hz
            cap = self.chip_l2_bw_bytes_per_sec / C
            return min(raw, cap)
        elif bus == "l1_to_l0a":
            return self.l1_to_l0a_per_cycle * self.freq_hz
        elif bus == "l1_to_l0b":
            return self.l1_to_l0b_per_cycle * self.freq_hz
        elif bus == "l0c_to_out":
            return self.l0c_to_out_per_cycle * self.freq_hz
        elif bus == "ub_to_l1":
            return self.ub_to_l1_per_cycle * self.freq_hz
        elif bus == "ub_to_reg":
            return self.ub_to_reg_per_cycle * self.freq_hz
        return 0.0

    def effective_load_bw(self, active_cores: int, reuse_count: int = 1) -> float:
        """Effective per-core bandwidth mixing HBM (cold) and L2 (warm reuse).

        First access uses HBM, subsequent (reuse_count - 1) use L2.
        Returns weighted average in bytes/sec.
        """
        if reuse_count <= 1:
            return self.per_core_bw("hbm", active_cores)
        hbm = self.per_core_bw("hbm", active_cores)
        l2 = self.per_core_bw("l2", active_cores)
        return 1.0 / (1.0 / reuse_count / hbm + (reuse_count - 1.0) / reuse_count / l2)

    def fixpipe_bw(self, active_cores: int, dest: str = "UB") -> float:
        """FixPipe effective bandwidth based on output destination.

        dest="UB"/"L1": core-local l0c_to_out peak (no cross-core contention).
        dest="GM":      capped by chip HBM write bandwidth with contention
                         (required for Split-KN FP32 partial sums written to GM
                         for V-core accumulation).
        """
        local_peak = self.per_core_bw("l0c_to_out", active_cores)
        if dest == "GM":
            C = max(active_cores, 1)
            hbm_cap = self.chip_hbm_bw_bytes_per_sec / C if self.chip_hbm_bw_bytes_per_sec > 0 else local_peak
            return min(local_peak, hbm_cap) if local_peak > 0 else hbm_cap
        return local_peak

    def mmad_throughput(self, dtype_size: int) -> float:
        """MMAD throughput in ops/sec for one core (each op = 1 MAC = 2 FLOPs)."""
        if dtype_size == 1:
            return self.mmad_ops_per_cycle_fp8 * self.freq_hz
        return self.mmad_ops_per_cycle_fp16 * self.freq_hz


ASCEND_910B = AscendHWSpec()

ASCEND_950 = AscendHWSpec(
    name="Ascend950",
    aic_num=32,
    aiv_num=64,
    # Chip-total fields (derived from per-cycle specs for backward compat)
    cube_tflops_bf16=4096 * 2 * 1.65e9 * 32 / 1e12,   # ~432.5 TFLOPS
    cube_tops_int8=4096 * 2 * 1.65e9 * 32 / 1e12,
    cube_tflops_fp8=8192 * 2 * 1.65e9 * 32 / 1e12,     # ~865.1 TFLOPS
    hbm_bw_bytes_per_sec=1.6e12,
    l2_bw_bytes_per_sec=5.2e12,
    fixpipe_bw_bytes_per_sec=256.0 * 1.65e9 * 32,       # l0c_to_out per core * cores
    vector_ops_per_sec_bf16=20.0e12,
    mte2_bw_bytes_per_sec=1.6e12,
    mte3_bw_bytes_per_sec=1.6e12,
    sync_overhead_us=0.0,
    scatter_efficiency=0.4,
    # Per-cycle specs
    freq_ghz=1.65,
    cache=CacheSpec(
        ub_size=256 * 1024,
        l1_size=512 * 1024,
        l0a_size=64 * 1024,
        l0b_size=256 * 1024,
        l0c_size=256 * 1024,
    ),
    hbm_bw_per_cycle=33.3,
    l2_bw_per_cycle=108.0,
    l1_to_l0a_per_cycle=256.0,
    l1_to_l0b_per_cycle=256.0,
    l0c_to_out_per_cycle=256.0,
    ub_to_l1_per_cycle=245.0,
    ub_to_reg_per_cycle=512.0,
    vec_instr_bytes_per_cycle=256.0,
    vec_ipc=1.3,
    mmad_ops_per_cycle_fp8=8192,
    mmad_ops_per_cycle_fp16=4096,
    chip_hbm_bw_bytes_per_sec=1.6e12,
    chip_l2_bw_bytes_per_sec=5.2e12,
)

CHIP_REGISTRY = {"910b": ASCEND_910B, "950": ASCEND_950}
