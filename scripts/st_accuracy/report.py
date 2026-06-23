from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from .st_cases import CaseResult
from .timing import TimingCollector


class ResultReporter:
    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)

    def write(self, results: Iterable[CaseResult], timings: TimingCollector | None = None) -> None:
        start = time.perf_counter()
        result_list = list(results)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with (self.out_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for result in result_list:
                handle.write(json.dumps(result.to_json_dict(), sort_keys=True) + "\n")

        counts = Counter(result.status for result in result_list)
        summary = {"total": len(result_list)}
        for status in ("passed", "failed", "error", "unsupported", "skipped"):
            summary[status] = counts.get(status, 0)
        if timings is not None:
            timings.add("report_write_ms", (time.perf_counter() - start) * 1000.0)
            summary["timings"] = timings.to_json_dict()
        with (self.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
