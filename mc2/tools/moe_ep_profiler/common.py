from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence


A5_FALLBACK_FREQ_MHZ = 1650.0


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    by_normalized = {normalized_name(name): name for name in fieldnames if name}
    for candidate in candidates:
        found = by_normalized.get(normalized_name(candidate))
        if found is not None:
            return found
    return None


def parse_number(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NONE", "-"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def iter_named_files(root: Path, prefixes: Sequence[str]) -> Iterator[Path]:
    normalized_prefixes = tuple(normalized_name(item) for item in prefixes)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stem = normalized_name(path.stem)
        if any(stem.startswith(prefix) for prefix in normalized_prefixes):
            yield path


def _frequency_to_mhz(value: float) -> float:
    # platform_config and msprof releases have used MHz, kHz and Hz.
    if value >= 10_000_000:
        return value / 1_000_000.0
    if value >= 10_000:
        return value / 1_000.0
    return value


def resolve_frequency_mhz(search_root: Path) -> dict[str, object]:
    """Resolve frequency using Current Freq, vec_freq, then the A5 fallback."""
    candidates: list[tuple[float, Path, str]] = []
    for path in iter_named_files(search_root, ["OpBasicInfo"]):
        try:
            rows = read_csv_rows(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        if not rows:
            continue
        key = find_column(rows[0].keys(), ["Current Freq", "Current Frequency"])
        if key is None:
            continue
        for row in rows:
            value = parse_number(row.get(key))
            if value is not None and value > 0:
                candidates.append((_frequency_to_mhz(value), path, key))
    if candidates:
        values = [item[0] for item in candidates]
        return {
            "mhz": percentile(values, 0.5),
            "source": "msprof.OpBasicInfo.Current Freq",
            "source_files": sorted({str(item[1]) for item in candidates}),
            "observed_mhz": values,
        }

    config_roots: list[Path] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        config_roots.append(Path(ascend_home))
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit_home:
        config_roots.append(Path(toolkit_home))
    seen: set[Path] = set()
    for root in config_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.ini"):
            if "platform_config" not in str(path).lower() or path in seen:
                continue
            seen.add(path)
            try:
                section = ""
                for raw_line in path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    line = raw_line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1].strip()
                        continue
                    if normalized_name(section) != "vectorcorespec" or "=" not in line:
                        continue
                    key, raw_value = line.split("=", 1)
                    if normalized_name(key) != "vecfreq":
                        continue
                    value = parse_number(raw_value)
                    if value is not None and value > 0:
                        return {
                            "mhz": _frequency_to_mhz(value),
                            "source": "platform_config.[VectorCoreSpec].vec_freq",
                            "source_files": [str(path)],
                            "observed_mhz": [_frequency_to_mhz(value)],
                        }
            except OSError:
                continue

    return {
        "mhz": A5_FALLBACK_FREQ_MHZ,
        "source": "A5 fallback",
        "source_files": [],
        "observed_mhz": [A5_FALLBACK_FREQ_MHZ],
    }
