# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Stage an existing frozen bundle; this command never imports operator source."""
import argparse
from pathlib import Path
import cannbot_framework as framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-opp", required=True, type=Path)
    args = parser.parse_args()
    result = framework.stage(
        bundle=args.bundle,
        export=framework.AclnnExportSpec(operator="FlashKda", public_name="FlashKda",
            l0_name="FlashKda", op_type="FlashKda", vendor="custom_transformer"),
        parent_opp=args.parent_opp,
        output=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
