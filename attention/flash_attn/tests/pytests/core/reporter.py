"""终端表格 + CSV 报告。"""

import csv
from pathlib import Path
from typing import Dict, List


class Reporter:
    """收集每条 case 的结果，支持中间统计和最终汇总输出。"""

    def __init__(self):
        self._results: Dict[str, dict] = {}

    def record(self, name: str, result: dict):
        self._results[name] = result

    def print_interim(self, total: int):
        self._print_table(self._results, total, is_final=False)

    def print_final(self) -> int:
        return self._print_table(self._results, len(self._results), is_final=True)

    def save_csv(self, path: Path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, ["case", "attn_pass", "attn_max_abs",
                                   "attn_fail_ratio", "lse_pass", "lse_max_abs",
                                   "lse_fail_ratio"])
            w.writeheader()
            for name, res in self._results.items():
                a = res.get("attn", {})
                l = res.get("lse", {})
                w.writerow({
                    "case": name,
                    "attn_pass": a.get("passed", False),
                    "attn_max_abs": a.get("max_abs", ""),
                    "attn_fail_ratio": a.get("fail_ratio", ""),
                    "lse_pass": l.get("passed", False),
                    "lse_max_abs": l.get("max_abs", ""),
                    "lse_fail_ratio": l.get("fail_ratio", ""),
                })

    @staticmethod
    def _print_table(results: dict, total: int, is_final: bool) -> int:
        SEP = "─" * 120
        max_len = max((len(n) for n in results.keys()), default=28)
        title = "  汇总结果" if is_final else f"  中间统计  ({len(results)}/{total})"
        print(f"\n┌{SEP}┐")
        print(f"│{title}")
        print(f"├{SEP}┤")
        hdr = (f"│  {'Case':<{max_len}}  "
               f"{'Attn':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │  "
               f"{'LSE':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │")
        print(hdr)
        print(f"├{SEP}┤")
        pass_cnt = fail_cnt = 0
        for name, res in results.items():
            a = res.get("attn", {})
            l = res.get("lse", {})
            a_tag = "PASS" if a.get("passed") else "FAIL"
            l_tag = ("PASS" if l.get("passed") else "FAIL") if l.get("total", 0) > 0 else "---"
            a_max = f"{a.get('max_abs', 0):.6f}" if a.get("total", 0) > 0 else "N/A"
            a_fr = f"{a.get('fail_ratio', 0)*100:.4f}%" if a.get("total", 0) > 0 else "N/A"
            l_max = f"{l.get('max_abs', 0):.6f}" if l.get("total", 0) > 0 else "-"
            l_fr = f"{l.get('fail_ratio', 0)*100:.4f}%" if l.get("total", 0) > 0 else "-"
            print(f"│  {name:<{max_len}}  "
                  f"{a_tag:>8}  {a_max:>12}  {a_fr:>10}  │  "
                  f"{l_tag:>8}  {l_max:>12}  {l_fr:>10}  │")
            if a.get("passed") and l.get("passed"): pass_cnt += 1
            else: fail_cnt += 1
        print(f"├{SEP}┤")
        label = f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(results)}"
        print(label)
        print(f"└{SEP}┘")
        return fail_cnt
