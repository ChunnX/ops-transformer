"""终端表格 + CSV 报告。"""

import csv
from pathlib import Path
from typing import Dict


class Reporter:
    """收集每条 case 的结果，支持中间统计和最终 CSV 输出。"""

    def __init__(self):
        self._results: Dict[str, dict] = {}
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None

    def set_csv(self, path: Path):
        """设置增量 CSV 输出路径，每条 record 自动追加。"""
        self._csv_path = path
        self._csv_file = open(path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            ["case", "dtype", "attn_pass", "attn_max_abs", "attn_fail_ratio",
             "lse_pass", "lse_max_abs", "lse_fail_ratio", "error"])
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def record(self, name: str, result: dict, error: str = None, dtype: str = ""):
        if error:
            result["_error"] = error
        if dtype:
            result["_dtype"] = dtype
        self._results[name] = result

        if self._csv_writer:
            a = result.get("attn", {})
            l = result.get("lse", {})
            self._csv_writer.writerow({
                "case": name,
                "dtype": dtype,
                "attn_pass": a.get("passed"),
                "attn_max_abs": a.get("max_abs", ""),
                "attn_fail_ratio": _pct(a.get("fail_ratio")),
                "lse_pass": l.get("passed") if l.get("total", 0) > 0 else "",
                "lse_max_abs": l.get("max_abs", ""),
                "lse_fail_ratio": _pct(l.get("fail_ratio")) if l.get("total", 0) > 0 else "",
                "error": error or "",
            })
            self._csv_file.flush()

    def print_interim(self, total: int):
        self._print_table(self._results, total, is_final=False)

    def print_final(self) -> int:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        return self._print_table(self._results, len(self._results), is_final=True)


def _pct(v) -> str:
    if v is None or v == float('nan') or v < 0:
        return ""
    return f"{v * 100:.4f}%"
