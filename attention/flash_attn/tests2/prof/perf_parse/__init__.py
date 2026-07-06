def parse_perf_summary(prof_dir: str) -> dict:
    raise NotImplementedError


def compare_with_baseline(result: dict, baseline: dict, threshold: float = 8.0) -> dict:
    raise NotImplementedError
