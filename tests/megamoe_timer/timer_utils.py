from pathlib import Path

import numpy as np
import torch


FIXED_TIMING_COUNT = 10
DYNAMIC_TYPE_COUNT = 9
MAX_DYNAMIC_ITER = 1024
CORE_COUNT = 96


def mega_moe_timer_count() -> int:
    timing_per_core = (
        (FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT * MAX_DYNAMIC_ITER) * 2 + 15
    ) // 16 * 16
    dynamic_iter_per_core = (DYNAMIC_TYPE_COUNT + 15) // 16 * 16
    return (timing_per_core + dynamic_iter_per_core) * CORE_COUNT


def create_mega_moe_timer() -> torch.Tensor:
    return torch.zeros(mega_moe_timer_count(), dtype=torch.int64).npu()


def save_mega_moe_timer(timer: torch.Tensor, rank: int, output_dir: str = "timer") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"rank_{rank}_npu_time.csv"
    np.savetxt(csv_path, timer.detach().cpu().numpy().reshape(-1, 1), delimiter=",", fmt="%d")
    return csv_path
