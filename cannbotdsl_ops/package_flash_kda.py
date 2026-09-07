# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Package the installed FlashKDA frozen OPP using an explicit Makeself tool."""
from pathlib import Path
import argparse
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--makeself", type=Path, required=True)
    args = parser.parse_args()
    tool = args.makeself.resolve(strict=True)
    payload = args.payload.resolve(strict=True)
    if not (payload / "install.sh").is_file() or not (payload / "packages").is_dir():
        raise ValueError("payload must be the parent CMake installation")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="cannbot-parent-makeself-") as temp:
        header = Path(temp) / "header.sh"
        content = tool.with_name("makeself-header.sh").read_text()
        # Huawei's Makeself header logs under HOME by default. Keep acceptance
        # logs under an explicit caller directory without changing HOME.
        content = content.replace('info_record_path="\\$HOME/log/makeself"',
            'info_record_path="\\${CANNBOT_PACKAGE_LOG_DIR:-\\$USER_PWD/.cannbotdsl-install-logs}"')
        if 'scriptargs="--\\$name_of_file"' in content:
            # Restore standard startup argv and verification-only --check.
            # The vendor header otherwise injects archive/cwd pseudo-options.
            content = "\n".join(line for line in content.split("\n")
                                if not line.startswith('scriptargs="--\\$name_of_file"'))
            content = content.replace('scriptargs="\\$scriptargs \\$1"\n    shift', 'exit 0', 1)
            content = content.replace('\t--noexec)', '    --)\n    shift\n    break\n    ;;\n\t--noexec)', 1)
        header.write_text(content)
        subprocess.run(["bash", str(tool), "--header", str(header), "--nox11", "--sha256", "--gzip",
            str(payload), str(output), "FlashKDA schema 2 frozen operator", "./install.sh"], check=True)
    print(output)


if __name__ == "__main__":
    main()
