"""Public batch execution interface — zero torch dependency."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import time
from typing import Any, Callable, List, Optional

from tqdm import tqdm

from npukernelbench.execution.process_pool import BatchProcessConfig, ProcessManager
from npukernelbench.execution.task_queue import TaskQueue


def batch_execute(
    tasks: List[Any],
    worker: Callable[..., Any],
    *,
    num_processes: int = 100,
    timeout: int = 300,
    other_args: Optional[dict] = None,
    desc: str = "",
    on_complete: Optional[Callable[[Any, Any, int], None]] = None,
    on_progress: Optional[Callable[[int], None]] = None,
) -> List[Any]:
    """Execute *worker(task, idx, **other_args)* in parallel subprocesses.

    Args:
        tasks:      List of task objects to process.
        worker:     Callable executed in a child process.
        num_processes: Max concurrent processes.
        timeout:    Seconds before killing a stuck process.
        other_args: Extra keyword args forwarded to *worker*.
        desc:       tqdm description.
        on_complete: Called as on_complete(task, result, idx) in the main process.
        on_progress: Called as on_progress(completed_count) after each task finishes.
    """
    mp.set_start_method("spawn", force=True)
    if not tasks:
        return []

    if other_args is None:
        other_args = {}

    cfg = BatchProcessConfig(
        max_processes=num_processes,
        timeout_seconds=timeout,
    )
    cfg.adjust_for_system()
    print(f"[BatchExecutor] config: {json.dumps(cfg.to_dict(), indent=2)}")

    queue = TaskQueue(tasks, max_retries=cfg.max_retries)
    pm = ProcessManager(cfg)
    results: List[Any] = [None] * len(tasks)
    completed = 0

    try:
        with tqdm(total=len(tasks), desc=desc) as pbar:
            while queue.has_pending() or pm.active_count > 0:
                # Launch new processes
                while queue.has_pending():
                    can, reason = pm.can_start(queue.pending_count())
                    if not can:
                        if "resource" in reason.lower():
                            time.sleep(cfg.resource_recovery_time)
                        break

                    item = queue.get_next()
                    if item is None:
                        break
                    idx, task = item

                    if pm.active_count > 0:
                        time.sleep(pm.start_delay)

                    ok = pm.start(idx, worker, task, other_args, cfg.to_dict())
                    if not ok:
                        queue.put_back(idx, task)
                        time.sleep(cfg.resource_recovery_time)
                        break

                # Poll active processes
                for idx in pm.active_indices():
                    status, result = pm.check_status(idx)

                    if status == "running":
                        continue

                    done = _finalize(idx, result, status, results, queue, tasks, on_complete, pbar)
                    if done:
                        completed += 1
                        if on_progress:
                            on_progress(completed)
                        if completed % cfg.gc_interval == 0:
                            gc.collect()

                    pm.cleanup(idx, force=(status == "timeout"))

                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted, cleaning up...")
        pm.cleanup_all()
        raise
    finally:
        pm.cleanup_all()

    return results


def _finalize(idx, result, status, results, queue, tasks, callback, pbar) -> bool:
    """Process a finished task. Returns True if task is done (no retry)."""
    if status == "completed":
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            if queue.should_retry(idx) and isinstance(result[1], (MemoryError, OSError)):
                queue.increment_retry(idx)
                queue.queue_for_retry(idx)
                return False
            results[idx] = result[1]
        else:
            results[idx] = result
    elif status in ("timeout", "crashed", "error"):
        if queue.should_retry(idx):
            queue.increment_retry(idx)
            queue.queue_for_retry(idx)
            return False
        results[idx] = result
    else:
        results[idx] = RuntimeError(f"Unknown status: {status}")

    if callback:
        try:
            callback(tasks[idx], results[idx], idx)
        except Exception as e:
            print(f"[WARNING] Callback error for task {idx}: {e}")

    pbar.update(1)
    return True
