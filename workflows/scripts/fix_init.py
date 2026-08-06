import os
import glob
import re

ROOT = r"D:\code\ops-transformer"

def fix_init(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    modified = content

    # Fix 1: Remove non-existent convert_<op> imports, replace with module import
    # Pattern: from .graph_convert_xxx import convert_xxx
    # Replace with: from . import graph_convert_xxx
    pattern = re.compile(r"from \.(graph_convert_\w+) import convert_\w+")
    modified = pattern.sub(r"from . import \1", modified)

    # Fix 2: Remove convert_<op> from __all__
    modified = re.sub(r',?\s*"convert_\w+"', '', modified)

    # Fix 3: Fix moe_distribute_dispatch - import from deep_ep
    if "from .moe_distribute_dispatch import moe_distribute_dispatch" in modified:
        modified = modified.replace(
            "from .moe_distribute_dispatch import moe_distribute_dispatch",
            "from .deep_ep import moe_distribute_dispatch_op_builder as moe_distribute_dispatch"
        )
        # Also add deep_ep module import for side effects
        if "from . import deep_ep" not in modified and "from .deep_ep import" not in modified:
            modified = modified.replace(
                'from . import graph_convert_moe_distribute_dispatch',
                'from . import deep_ep\nfrom . import graph_convert_moe_distribute_dispatch'
            )

    # Fix 4: moe_distribute_combine __init__ - also imports from deep_ep which is in another dir
    if "moe_distribute_combine/torch_extension/__init__.py" in filepath:
        modified = (
            "from cann_ops_transformer.ops.mc2.moe_distribute_dispatch.deep_ep import moe_distribute_combine_op_builder\n"
        )

    # Clean up double empty lines
    while "\n\n\n" in modified:
        modified = modified.replace("\n\n\n", "\n\n")

    if modified != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(modified)
        return True
    return False


def main():
    pattern = os.path.join(ROOT, "*", "*", "torch_extension", "__init__.py")
    files = sorted(glob.glob(pattern))

    for f in files:
        rel = os.path.relpath(f, ROOT)
        if fix_init(f):
            print(f"FIXED: {rel}")
        else:
            print(f"  OK: {rel}")


if __name__ == "__main__":
    main()
