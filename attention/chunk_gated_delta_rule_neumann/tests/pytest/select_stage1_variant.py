#!/usr/bin/env python3
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parents[1]
OP_KERNEL_DIR = OP_DIR / "op_kernel"
SELECTED_DIR = OP_KERNEL_DIR / "selected"
STAGE1_SELECTED = SELECTED_DIR / "stage1_selected.h"
STAGE1_CONFIG = SELECTED_DIR / "stage1_selected_config.h"

VARIANTS = {
    "C16_FP32": 16,
    "C16_FP16": 16,
    "C64_1616_FP32": 64,
    "C64_1616_FP16": 64,
    "C64_3232_FP32": 64,
    "C64_3232_FP16": 64,
}


def header_for(suite, variant):
    return f"../{suite}/chunk_gated_delta_rule_neumann_stage1_{variant}_{suite}.h"


def write_selected(suite, variant):
    if suite not in {"baseline", "halflocal", "fulllocal"}:
        raise ValueError(f"unknown suite: {suite}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    header = OP_KERNEL_DIR / suite / f"chunk_gated_delta_rule_neumann_stage1_{variant}_{suite}.h"
    if not header.exists():
        raise FileNotFoundError(header)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_SELECTED.write_text(
        "\n".join(
            [
                "#ifndef CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_H",
                "#define CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_H",
                "",
                f'#include "{header_for(suite, variant)}"',
                "",
                "#endif  // CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_H",
                "",
            ]
        ),
        encoding="utf-8",
    )
    STAGE1_CONFIG.write_text(
        "\n".join(
            [
                "#ifndef CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_CONFIG_H",
                "#define CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_CONFIG_H",
                "",
                f'#define CGDR_NEUMANN_SELECTED_SUITE "{suite}"',
                f'#define CGDR_NEUMANN_SELECTED_VARIANT "{variant}"',
                f"#define CGDR_NEUMANN_SELECTED_CHUNK_SIZE {VARIANTS[variant]}",
                "",
                "#endif  // CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_SELECTED_CONFIG_H",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"selected suite={suite} variant={variant} chunk={VARIANTS[variant]}")


def main():
    parser = argparse.ArgumentParser(description="Select ChunkGatedDeltaRuleNeumann Stage1 variant.")
    parser.add_argument("--suite", required=True, choices=["baseline", "halflocal", "fulllocal"])
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    args = parser.parse_args()
    write_selected(args.suite, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
