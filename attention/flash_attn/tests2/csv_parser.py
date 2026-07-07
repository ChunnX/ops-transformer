import csv

CSV_HEADERS = ["case", "dtype", "key", "pass", "max_abs", "mean_abs", "fail_ratio", "error"]


def parse_csv(csv_path: str) -> list[dict]:
    """解析CSV文件, 返回dict列表"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
