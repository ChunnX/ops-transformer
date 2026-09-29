# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Export a source-only dependency cache for a disconnected build machine."""

import argparse
from pathlib import Path
import tarfile
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "sim_build")
    parser.add_argument("--dependencies", type=Path, default=ROOT / "sim_deps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deps = args.dependencies.resolve()
    cmake = args.build_dir / "native/prepare_build/_deps/cann-cmake-src"
    if not (deps / "opbase").exists() or not (deps / "ops-tensor").exists():
        raise SystemExit(
            "Run ./mega-sim build --host-only to prepare dependencies first"
        )
    if (
        not (deps / "cann-cmake").exists()
        and not (cmake / "function/prepare.cmake").exists()
    ):
        raise SystemExit("Missing pinned cann-cmake source")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    def keep_source(member):
        parts = Path(member.name).parts
        if "__pycache__" in parts or "lib_cache" in parts:
            return None
        return member

    with tarfile.open(args.output, "w:gz") as archive:
        archive.add(deps, arcname="sim_deps", filter=keep_source)
        if not (deps / "cann-cmake").exists():
            archive.add(cmake, arcname="sim_deps/cann-cmake", filter=keep_source)
        # cann-cmake's JSON package lookup uses the dependency root/json directory.
        for path in (deps / "pkg").glob("json-*.tar.gz"):
            archive.add(path, arcname="sim_deps/" + path.name)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
