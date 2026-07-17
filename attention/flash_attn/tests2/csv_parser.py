"""CSV 读写工具。

parse_csv 是通用 CSV 解析, 与测试框架无耦合, 可独立复用;
CSV_HEADERS 是精度报告的列契约, 供 Reporter 写出 / parse_csv 读回对齐。
"""
import csv

CSV_HEADERS = ["case", "dtype", "key", "pass", "max_abs", "mean_abs", "fail_ratio", "error"]


def parse_csv(csv_path: str) -> list[dict]:
    """解析CSV文件, 返回dict列表"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
