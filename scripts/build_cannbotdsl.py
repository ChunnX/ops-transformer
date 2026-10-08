#!/usr/bin/env python3
"""Generate Native providers and ACLNN sources with the installed OpKit wheel."""
import argparse
from pathlib import Path
import opkit
from opkit.integrity import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu-arch", default="dav-3510")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    result = opkit.build(collectors=source.parent, pattern=source.name,
                         target=args.npu_arch, output=args.output)
    verify_manifest(result)
    print(result)


if __name__ == "__main__":
    main()
