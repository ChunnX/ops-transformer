# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Generate the A2/A3 TurboQuant registration without changing baseline JSON."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("kernel_name")
    args = parser.parse_args()
    op_name = "MixedQuantSparseFlashMlaMetadata"
    metadata = json.loads(args.source.read_text(encoding="utf-8"))[op_name]
    metadata["opInfo"]["kernelSo"] = args.kernel_name
    args.destination.write_text(json.dumps({op_name: metadata}, indent=4) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
