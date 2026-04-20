"""Warmup + timing utilities for performance measurement."""

from __future__ import annotations


class PerformanceTimer:
    """Warmup model and measure average execution time on NPU."""

    def __init__(self, device):
        self.device = device

    def warmup_and_measure(self, model, inputs, num_trials: int) -> float:
        """Returns average ms per inference."""
        import torch

        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize(self.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(*inputs)
        end.record()
        torch.cuda.synchronize(self.device)

        return start.elapsed_time(end) / num_trials
