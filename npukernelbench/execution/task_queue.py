"""Task queue with retry logic."""

from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Tuple


class TaskQueue:
    """Manages task queue with priority for retries."""

    def __init__(self, tasks: List[Any], max_retries: int = 1):
        self.tasks = tasks
        self.max_retries = max_retries
        self.pending: deque[Tuple[int, Any]] = deque(enumerate(tasks))
        self.retry_queue: deque[Tuple[int, Any]] = deque()
        self._retry_counts: dict[int, int] = {}

    def has_pending(self) -> bool:
        return bool(self.pending or self.retry_queue)

    def get_next(self) -> Optional[Tuple[int, Any]]:
        if self.retry_queue:
            return self.retry_queue.popleft()
        if self.pending:
            idx, task = self.pending.popleft()
            self._retry_counts[idx] = 0
            return idx, task
        return None

    def put_back(self, idx: int, task: Any) -> None:
        if self._retry_counts.get(idx, 0) > 0:
            self.retry_queue.appendleft((idx, task))
        else:
            self.pending.appendleft((idx, task))

    def should_retry(self, idx: int) -> bool:
        return self._retry_counts.get(idx, 0) < self.max_retries

    def increment_retry(self, idx: int) -> int:
        self._retry_counts[idx] = self._retry_counts.get(idx, 0) + 1
        return self._retry_counts[idx]

    def queue_for_retry(self, idx: int) -> None:
        if idx < len(self.tasks):
            self.retry_queue.append((idx, self.tasks[idx]))

    def pending_count(self) -> int:
        return len(self.pending) + len(self.retry_queue)
