"""System resource monitoring — zero torch dependency."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Tuple

import psutil

# BatchProcessConfig is in process_pool.py — we accept a dict, no circular import


def get_system_thread_stats() -> Tuple[int, int, float]:
    """Return (active_threads, total_threads, load_avg)."""
    try:
        total = active = 0
        for proc in psutil.process_iter(["num_threads", "status"]):
            try:
                info = proc.info
                total += info["num_threads"]
                if info["status"] in (psutil.STATUS_RUNNING, psutil.STATUS_DISK_SLEEP):
                    active += info["num_threads"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        return active, total, load
    except Exception:
        return 0, 0, 0.0


def check_system_resources(cfg: dict) -> Tuple[bool, str]:
    """Check memory, CPU, thread limits. Returns (ok, reason)."""
    try:
        mem = psutil.virtual_memory()
        if mem.used / (1024 ** 3) > cfg["memory_threshold_gb"]:
            return False, f"Memory usage too high: {mem.used / (1024**3):.1f}GB"

        if cfg.get("use_thread_aware_scaling", False):
            active, total, load = get_system_thread_stats()
            if total > cfg.get("total_thread_threshold", 20000):
                return False, f"Total thread count too high: {total}"
            if active > cfg.get("active_thread_threshold", 5000):
                return False, f"Active thread count too high: {active}"

        cpu = psutil.cpu_percent(interval=0.1)
        if cpu > cfg["cpu_threshold_percent"]:
            return False, f"CPU usage too high: {cpu:.1f}%"

        return True, "OK"
    except Exception as e:
        return False, f"Resource check failed: {e}"


class ResourceMonitor:
    """Periodically check system resources for process scaling decisions."""

    def __init__(self, cfg_dict: dict):
        self._cfg = cfg_dict
        self._last_check = 0.0
        self._history: deque[Tuple[float, bool]] = deque(maxlen=10)

    def check(self) -> Tuple[bool, str]:
        now = time.time()
        if now - self._last_check < self._cfg.get("resource_check_interval", 5.0):
            return True, "OK"
        self._last_check = now
        ok, reason = check_system_resources(self._cfg)
        self._history.append((now, ok))
        return ok, reason

    def optimal_count(self, current: int, pending: int) -> int:
        """Suggest optimal number of concurrent processes."""
        ok, _ = self.check()
        if not ok:
            cpu = psutil.cpu_percent(interval=0.1)
            if cpu >= 30:
                recent_fails = sum(1 for _, s in self._history if not s)
                if recent_fails > len(self._history) * 0.7:
                    return max(self._cfg.get("min_processes", 1), current // 2)
            return current

        if pending > current * 2:
            return min(self._cfg.get("max_processes", 200), current + max(1, pending // 10))
        return current
