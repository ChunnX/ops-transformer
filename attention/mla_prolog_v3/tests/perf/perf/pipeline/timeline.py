#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASCII art timeline renderer for the pipeline DAG result."""

from .dag import PipelineResult


def format_timeline(result: PipelineResult, width: int = 100) -> str:
    """Generate ASCII art timeline showing AIC/AIV stages."""
    total = result.total_us
    if total == 0:
        return "(no stages)"

    scale = (width - 20) / total  # chars per microsecond

    lines = []
    lines.append(f"Total: {total:.2f} us")
    lines.append(f"AIC busy: {result.aic_busy_us:.2f} us | AIV busy: {result.aiv_busy_us:.2f} us | Sync: {result.sync_overhead_us:.2f} us")
    lines.append("")

    # Time ruler
    ruler_marks = 5
    ruler = " " * 18 + "|"
    for i in range(1, ruler_marks + 1):
        t = total * i / ruler_marks
        pos = int(t * scale)
        ruler = ruler.ljust(18 + pos + 1)
        ruler = ruler[:18 + pos] + "|"
    lines.append(ruler)

    time_labels = " " * 18 + "0"
    for i in range(1, ruler_marks + 1):
        t = total * i / ruler_marks
        pos = int(t * scale)
        label = f"{t:.1f}"
        time_labels = time_labels.ljust(18 + pos + 1)
        time_labels = time_labels[:18 + pos - len(label) + 1] + label
    lines.append(time_labels + " us")
    lines.append("")

    # Group stages by pipe
    for pipe in ["AIC", "AIV"]:
        lines.append(f"  {pipe}:")
        pipe_stages = [
            s for s in result.stages.values()
            if s.pipe == pipe
            and not s.name.startswith("sync_")
            and not s.name.startswith("aic_wait")
            and not s.name.startswith("aiv_wait")
        ]
        pipe_stages.sort(key=lambda s: s.start_us)

        for s in pipe_stages:
            start_col = int(s.start_us * scale)
            end_col = int(s.end_us * scale)
            bar_len = max(end_col - start_col, 1)

            label = s.name[:14]
            cp_marker = "*" if s.on_critical_path else " "

            bar = " " * start_col + "[" + "=" * max(bar_len - 2, 0) + "]"
            line = f"  {cp_marker} {label:14s} {bar} {s.duration_us:.2f}us"
            lines.append(line)

        lines.append("")

    # Critical path
    lines.append("Critical path:")
    cp_stages = [result.stages[n] for n in result.critical_path
                 if not n.startswith("sync_") and "wait" not in n]
    cp_names = [f"{s.name}({s.duration_us:.1f})" for s in cp_stages]
    lines.append("  " + " -> ".join(cp_names))

    return "\n".join(lines)
