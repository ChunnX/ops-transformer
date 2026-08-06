import os
import re
import glob

ROOT = r"D:\code\ops-transformer"
pattern = os.path.join(ROOT, "*", "*", "torch_extension", "*.py")
files = [f for f in glob.glob(pattern) if os.path.basename(f) != "__init__.py" and "graph_convert" not in os.path.basename(f)]

for f in sorted(files):
    with open(f, "r", encoding="utf-8") as fp:
        content = fp.read()
    
    # Check for multi-line builder pattern: var = (\n    XxxOpBuilder()\n)
    # That is missing _ensure_initialized() on the next line
    m = re.search(r'(\w+_op_builder)\s*=\s*\(\s*\n\s+(\w+\(\))\s*\n\)', content)
    if m:
        lines = content.split('\n')
        # Find the line number of the closing paren
        for i, line in enumerate(lines):
            if line.strip() == ')' and i > 0 and '_op_builder' in lines[i-2]:
                # Check next non-empty line
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and '_ensure_initialized()' not in lines[j] and '.load()' not in lines[j]:
                    rel = os.path.relpath(f, ROOT)
                    print(f"MISSING: {rel} (var={m.group(1)}, line={i+1})")
                break
