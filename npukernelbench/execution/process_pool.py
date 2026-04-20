"""Process pool management — zero torch dependency."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from npukernelbench.execution.resource_monitor import ResourceMonitor, check_system_resources


class SegmentationFaultError(Exception):
    pass


class BatchProcessConfig:
    """Tunable parameters for the process pool."""

    def __init__(
        self,
        max_processes: int = 200,
        min_processes: int = 10,
        timeout_seconds: int = 300,
        max_retries: int = 1,
        threads_per_process: int = 8,
    ):
        self.max_processes = max_processes
        self.min_processes = min_processes
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.threads_per_process = threads_per_process
        self.process_start_delay = 0.01

        # Resource thresholds
        self.memory_threshold_gb = 1400.0
        self.cpu_threshold_percent = 99.0
        self.resource_check_interval = 5.0
        self.resource_recovery_time = 2.0
        self.use_thread_aware_scaling = True
        self.active_thread_threshold = 5000
        self.total_thread_threshold = 20000
        self.load_average_threshold: Optional[float] = None

        self.process_cleanup_timeout = 5.0
        self.gc_interval = 100

    def adjust_for_system(self) -> BatchProcessConfig:
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.memory_threshold_gb = min(self.memory_threshold_gb, mem_gb * 0.95)
        cpu_count = mp.cpu_count()
        self.load_average_threshold = cpu_count * 3.0
        self.active_thread_threshold = cpu_count * 200
        self.total_thread_threshold = cpu_count * 1000
        self.min_processes = max(1, int(self.max_processes * 0.1))
        return self

    def to_dict(self) -> dict:
        return {
            "max_processes": self.max_processes,
            "min_processes": self.min_processes,
            "memory_threshold_gb": self.memory_threshold_gb,
            "cpu_threshold_percent": self.cpu_threshold_percent,
            "resource_check_interval": self.resource_check_interval,
            "resource_recovery_time": self.resource_recovery_time,
            "use_thread_aware_scaling": self.use_thread_aware_scaling,
            "active_thread_threshold": self.active_thread_threshold,
            "total_thread_threshold": self.total_thread_threshold,
            "load_average_threshold": self.load_average_threshold,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class _ProcessInfo:
    process: mp.Process
    connection: mp.connection.Connection
    start_time: float
    task_idx: int


class ProcessManager:
    """Create, monitor, and clean up worker processes."""

    def __init__(self, cfg: BatchProcessConfig):
        self.cfg = cfg
        self._active: Dict[int, _ProcessInfo] = {}
        self.resource_monitor = ResourceMonitor(cfg.to_dict())
        self._start_delay = cfg.process_start_delay

    # --- public API --------------------------------------------------

    def can_start(self, pending: int) -> Tuple[bool, str]:
        current = len(self._active)
        if current >= self.resource_monitor.optimal_count(current, pending):
            return False, "Process limit reached"
        return check_system_resources(self.cfg.to_dict())

    def start(
        self,
        task_idx: int,
        worker: Callable,
        task_obj: Any,
        other_args: dict,
        config_dict: dict,
    ) -> bool:
        parent_conn, child_conn = mp.Pipe()
        try:
            proc = mp.Process(
                target=_process_wrapper,
                args=(_execute_task, (worker, task_obj, task_idx, other_args, config_dict), child_conn),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._active[task_idx] = _ProcessInfo(
                process=proc, connection=parent_conn, start_time=time.time(), task_idx=task_idx,
            )
            self._start_delay = max(0.001, self._start_delay * 0.95)
            return True
        except Exception as e:
            _safe_close(parent_conn)
            _safe_close(child_conn)
            self._start_delay = min(1.0, self._start_delay * 1.5)
            print(f"Failed to start process for task {task_idx}: {e}")
            return False

    def check_status(self, task_idx: int) -> Tuple[str, Any]:
        """Return (status, result_or_exception)."""
        info = self._active.get(task_idx)
        if info is None:
            return "not_found", None

        if time.time() - info.start_time > self.cfg.timeout_seconds:
            return "timeout", TimeoutError(f"Task {task_idx} timed out")

        if info.process.is_alive():
            if _safe_poll(info.connection):
                data = _safe_recv(info.connection)
                if data is not None:
                    _, result = data
                    return "completed", result
            return "running", None

        # Process finished
        if info.process.exitcode == 0:
            if _safe_poll(info.connection):
                data = _safe_recv(info.connection)
                if data is not None:
                    _, result = data
                    return "completed", result
            return "error", RuntimeError(f"No result for task {task_idx}")
        return "crashed", SegmentationFaultError(f"Exit code {info.process.exitcode} for task {task_idx}")

    def cleanup(self, task_idx: int, force: bool = False) -> None:
        info = self._active.pop(task_idx, None)
        if info is None:
            return
        _safe_close(info.connection)
        if info.process.is_alive():
            if force:
                _safe_kill(info.process)
            else:
                _safe_terminate(info.process, self.cfg.process_cleanup_timeout)

    def cleanup_all(self) -> None:
        for idx in list(self._active):
            self.cleanup(idx, force=True)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def active_indices(self) -> List[int]:
        return list(self._active)

    @property
    def start_delay(self) -> float:
        return self._start_delay


# --- internal helpers -----------------------------------------------

def _process_wrapper(target, args, conn):
    try:
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        signal.signal(signal.SIGINT, lambda *_: os._exit(0))
        result = target(*args)
        conn.send(("success", result))
    except Exception as e:
        conn.send(("error", e))
    finally:
        _safe_close(conn)


def _execute_task(worker, task_obj, idx, other_args, config_dict):
    return worker(task_obj, idx, **other_args)


def _safe_poll(conn) -> bool:
    try:
        return not conn.closed and conn.poll()
    except (OSError, ValueError, AttributeError):
        return False


def _safe_recv(conn):
    try:
        return conn.recv() if not conn.closed else None
    except (OSError, ValueError, EOFError):
        return None


def _safe_close(conn):
    if conn is not None and not conn.closed:
        try:
            conn.close()
        except Exception:
            pass


def _safe_terminate(proc, timeout):
    try:
        proc.terminate()
        proc.join(timeout)
        if proc.is_alive():
            _safe_kill(proc)
    except Exception:
        pass


def _safe_kill(proc):
    try:
        proc.kill()
    except Exception:
        pass
