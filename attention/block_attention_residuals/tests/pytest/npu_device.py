# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""NPU 设备选择：测试前解析 npu-smi，优先选用空闲卡。"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NpuStatus:
    device_id: int
    aicore_pct: int = 0
    hbm_used_mb: int = 0
    hbm_total_mb: int = 65536
    has_process: bool = False

    @property
    def is_idle(self) -> bool:
        return not self.has_process and self.aicore_pct == 0


def _parse_npu_smi_output(text: str) -> List[NpuStatus]:
    devices: Dict[int, NpuStatus] = {}
    in_process_section = False
    pending_npu_id: Optional[int] = None

    # NPU 汇总行：| 0     910B3  | OK | 90.8 | ...
    npu_header_re = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|\s*\S+\s+\|\s*(\d+(?:\.\d+)?)")
    # 芯片明细行：| 0  | 0000:... | AICore% | ... | HBM_used / HBM_total |
    chip_detail_re = re.compile(
        r"^\|\s*(\d+)\s+\|[^|]+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)"
    )
    idle_re = re.compile(r"No running processes found in NPU\s+(\d+)")
    busy_re = re.compile(r"^\|\s*(\d+)\s+\d+\s+\|\s*(\d+)\s+\|")

    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_section = True
            continue

        if not in_process_section:
            m_hdr = npu_header_re.match(line.strip())
            if m_hdr:
                pending_npu_id = int(m_hdr.group(1))
                continue

            m_chip = chip_detail_re.match(line.strip())
            if m_chip and pending_npu_id is not None:
                dev_id = pending_npu_id
                pending_npu_id = None
                devices[dev_id] = NpuStatus(
                    device_id=dev_id,
                    aicore_pct=int(m_chip.group(2)),
                    hbm_used_mb=int(m_chip.group(3)),
                    hbm_total_mb=int(m_chip.group(4)),
                )
            continue

        m_idle = idle_re.search(line)
        if m_idle:
            dev_id = int(m_idle.group(1))
            if dev_id in devices:
                devices[dev_id].has_process = False
            continue

        m_busy = busy_re.match(line.strip())
        if m_busy:
            dev_id = int(m_busy.group(1))
            if dev_id in devices:
                devices[dev_id].has_process = True

    return sorted(devices.values(), key=lambda x: x.device_id)


def query_npu_status(timeout_sec: int = 30) -> List[NpuStatus]:
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if proc.returncode != 0 and not proc.stdout:
        return []
    return _parse_npu_smi_output(proc.stdout)


def pick_free_device_id(statuses: Optional[List[NpuStatus]] = None) -> int:
    statuses = statuses if statuses is not None else query_npu_status()
    if not statuses:
        return int(os.environ.get("DEVICE_ID", "0"))

    idle = [s for s in statuses if s.is_idle]
    if idle:
        return min(idle, key=lambda s: s.hbm_used_mb).device_id

    return min(statuses, key=lambda s: (s.has_process, s.aicore_pct, s.hbm_used_mb)).device_id


def resolve_device_id(force_auto: bool = False) -> int:
    force_auto = force_auto or os.environ.get("FORCE_AUTO_DEVICE") == "1"
    if not force_auto and "DEVICE_ID" in os.environ:
        return int(os.environ["DEVICE_ID"])

    device_id = pick_free_device_id()
    os.environ["DEVICE_ID"] = str(device_id)
    return device_id


def format_status_summary(statuses: Optional[List[NpuStatus]] = None) -> str:
    statuses = statuses if statuses is not None else query_npu_status()
    if not statuses:
        return "npu-smi: no device info"
    lines = ["NPU status summary:"]
    for s in statuses:
        flag = "idle" if s.is_idle else ("busy" if s.has_process else "used")
        lines.append(
            f"  NPU {s.device_id}: AICore={s.aicore_pct}% HBM={s.hbm_used_mb}/{s.hbm_total_mb}MB [{flag}]"
        )
    return "\n".join(lines)
