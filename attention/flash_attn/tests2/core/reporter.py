import csv
from pathlib import Path

from reports.csv_parser import CSV_HEADERS


class Reporter:
    def __init__(self):
        self._results = {}
        self._dtypes = {}
        self._csv_file = None
        self._csv_writer = None

    def set_csv(self, path: Path):
        exists = path.exists()
        self._csv_file = open(path, "a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, CSV_HEADERS)
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def record(self, name: str, result: dict, error: str = None, dtype: str = ""):
        if error:
            result["error"] = error
        self._results[name] = result
        self._dtypes[name] = dtype
        if self._csv_writer:
            for key, m in result.items():
                if not isinstance(m, dict):
                    continue
                self._csv_writer.writerow({
                    "case": name,
                    "dtype": dtype,
                    "key": key,
                    "pass": m.get("passed"),
                    "max_abs": m.get("max_abs", ""),
                    "mean_abs": m.get("mean_abs", ""),
                    "fail_ratio": m.get("fail_ratio", ""),
                    "error": error or "",
                })
            self._csv_file.flush()

    def print_interim(self, total: int):
        self._print_table(self._results, self._dtypes, total, is_final=False)

    def print_final(self) -> int:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        return self._print_table(self._results, self._dtypes, len(self._results), is_final=True)

    @staticmethod
    def _print_table(results: dict, dtypes: dict, total: int, is_final: bool) -> int:
        max_len = max((len(n) for n in results), default=8)
        width = max_len + 52
        SEP = "─" * width
        title = "  汇总结果" if is_final else f"  中间统计  ({len(results)}/{total})"
        print(f"\n┌{SEP}┐")
        print(f"│{title}")
        print(f"├{SEP}┤")
        print(f"│  {'Case':<{max_len}}  {'Dtype':>6}  {'Keys':<28}  {'Status':>6}")
        print(f"├{SEP}┤")
        pass_cnt = fail_cnt = 0
        for name, res in results.items():
            tags = []
            all_pass = True
            for k, v in res.items():
                if not isinstance(v, dict):
                    continue
                p = v.get("passed")
                tags.append(f"{k}:{'P' if p else 'F'}")
                if not p:
                    all_pass = False
            status = "PASS" if all_pass else "FAIL"
            print(f"│  {name:<{max_len}}  {dtypes.get(name, ''):>6}  {' '.join(tags):<28}  {status:>6}")
            if all_pass:
                pass_cnt += 1
            else:
                fail_cnt += 1
        print(f"├{SEP}┤")
        print(f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(results)}")
        print(f"└{SEP}┘")
        return fail_cnt

    @staticmethod
    def check_results(result: dict) -> None:
        import pytest
        for key, m in result.items():
            if not isinstance(m, dict):
                continue
            if not m.get("passed"):
                total = m.get("total", 0)
                fail_cnt = m.get("fail_cnt", 0)
                ratio = m.get("fail_ratio", float("nan"))
                pass_rate = (total - fail_cnt) / total * 100 if total > 0 else 0.0
                pytest.fail(
                    f"精度校验失败 key={key}, pass率={pass_rate:.2f}%, fail_ratio={ratio}")


def make_failure_result(error: str) -> dict:
    return {"out": {"passed": False, "max_abs": float("nan"), "mean_abs": float("nan"),
                    "fail_cnt": 0, "total": 0, "fail_ratio": float("nan")}, "error": error}
