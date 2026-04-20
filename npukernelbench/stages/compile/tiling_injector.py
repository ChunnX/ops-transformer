"""Auto-inject tiling parameter prints into C++ source files.

Parses *_tiling.h to extract all TILING_DATA_FIELD_DEF entries, then injects
standardized OP_LOGD prints with [TILING] prefix after each SaveToBuffer call
in the corresponding tiling .cpp files.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field


FORMAT_MAP = {
    "uint64_t": "%lu",
    "int64_t": "%ld",
    "uint32_t": "%u",
    "int32_t": "%d",
    "uint16_t": "%u",
    "int16_t": "%d",
    "uint8_t": "%u",
    "int8_t": "%d",
    "float": "%f",
    "double": "%f",
    "bool": "%d",
}

OP_LOGD_GUARD = """\
#ifndef OP_LOGD
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\\n")
#endif
"""

_RE_SCALAR = re.compile(r"TILING_DATA_FIELD_DEF\(\s*(\w+)\s*,\s*(\w+)\s*\)")
_RE_ARRAY = re.compile(
    r"TILING_DATA_FIELD_DEF_ARR\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)"
)
_RE_STRUCT = re.compile(r"TILING_DATA_FIELD_DEF_STRUCT\(\s*(\w+)\s*,\s*(\w+)\s*\)")
_RE_CLASS = re.compile(r"BEGIN_TILING_DATA_DEF\(\s*(\w+)\s*\)")
_RE_SAVE_BUF = re.compile(r"(\w+)\.SaveToBuffer\(")
_RE_SET_DATA_SIZE = re.compile(r"SetDataSize\(")


@dataclass
class TilingField:
    name: str
    cpp_type: str
    is_array: bool = False
    array_size: int = 0


@dataclass
class TilingDef:
    class_name: str
    fields: list[TilingField] = field(default_factory=list)


def parse_tiling_header(header_path: str) -> TilingDef | None:
    """Parse a *_tiling.h file and extract all field definitions."""
    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()

    m = _RE_CLASS.search(content)
    if not m:
        return None
    class_name = m.group(1)

    fields: list[TilingField] = []
    for line in content.splitlines():
        line = line.strip()
        # Array fields
        m_arr = _RE_ARRAY.search(line)
        if m_arr:
            fields.append(TilingField(
                name=m_arr.group(3), cpp_type=m_arr.group(1),
                is_array=True, array_size=int(m_arr.group(2)),
            ))
            continue
        # Skip struct fields (too complex to auto-print)
        if _RE_STRUCT.search(line):
            continue
        # Scalar fields
        m_s = _RE_SCALAR.search(line)
        if m_s:
            fields.append(TilingField(name=m_s.group(2), cpp_type=m_s.group(1)))

    return TilingDef(class_name=class_name, fields=fields)


def _generate_print_block(fields: list[TilingField], var_name: str, indent: str) -> str:
    """Generate C++ OP_LOGD print statements for all tiling fields."""
    lines = [f"{indent}// === AUTO-INJECTED TILING DUMP BEGIN ==="]
    for f in fields:
        fmt = FORMAT_MAP.get(f.cpp_type, "%d")
        if f.is_array:
            lines.append(
                f'{indent}for (int _i = 0; _i < {f.array_size}; _i++) '
                f'{{ OP_LOGD(context, "[TILING] {f.name}[%d]: {fmt}.", _i, '
                f'({f.cpp_type}){var_name}.get_{f.name}()[_i]); }}'
            )
        else:
            lines.append(
                f'{indent}OP_LOGD(context, "[TILING] {f.name}: {fmt}.", '
                f'{var_name}.get_{f.name}());'
            )
    lines.append(f"{indent}// === AUTO-INJECTED TILING DUMP END ===")
    return "\n".join(lines)


def _inject_logd_macro(content: str) -> str:
    """Ensure OP_LOGD macro is defined. Insert after last #include if missing."""
    if "#define OP_LOGD" in content:
        return content
    lines = content.split("\n")
    last_include = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#include"):
            last_include = i
    insert_at = last_include + 1 if last_include >= 0 else 0
    lines.insert(insert_at, OP_LOGD_GUARD)
    return "\n".join(lines)


def _inject_into_cpp(cpp_path: str, fields: list[TilingField], logger=None) -> bool:
    """Find SaveToBuffer + SetDataSize and inject tiling prints after it."""
    with open(cpp_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already injected
    if "AUTO-INJECTED TILING DUMP BEGIN" in content:
        if logger:
            logger.debug(f"Tiling dump already injected in {cpp_path}")
        return False

    lines = content.split("\n")
    new_lines: list[str] = []
    injected = False
    i = 0
    while i < len(lines):
        new_lines.append(lines[i])

        # Detect SaveToBuffer
        m = _RE_SAVE_BUF.search(lines[i])
        if m and not injected:
            var_name = m.group(1)
            # Look ahead for SetDataSize (within next 5 lines)
            for j in range(i + 1, min(i + 6, len(lines))):
                if _RE_SET_DATA_SIZE.search(lines[j]):
                    # Add remaining lines up to and including SetDataSize
                    for k in range(i + 1, j + 1):
                        new_lines.append(lines[k])
                    # Inject print block after SetDataSize
                    indent = _guess_indent(lines[j])
                    block = _generate_print_block(fields, var_name, indent)
                    new_lines.append(block)
                    injected = True
                    i = j + 1
                    break
            else:
                i += 1
                continue
            continue

        i += 1

    if not injected:
        if logger:
            logger.warning(f"No SaveToBuffer + SetDataSize found in {cpp_path}")
        return False

    new_content = _inject_logd_macro("\n".join(new_lines))
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    if logger:
        logger.info(f"Injected tiling dump into {cpp_path}")
    return True


def _guess_indent(line: str) -> str:
    """Extract leading whitespace from a line."""
    return line[: len(line) - len(line.lstrip())]


def inject_tiling_prints(src_dir: str, logger=None) -> bool:
    """Scan src_dir for tiling headers and inject prints into corresponding cpp files.

    Returns True if any injection was performed.
    """
    host_dir = os.path.join(src_dir, "op_host")
    if not os.path.isdir(host_dir):
        if logger:
            logger.debug(f"No op_host directory in {src_dir}")
        return False

    headers = glob.glob(os.path.join(host_dir, "*_tiling.h"))
    if not headers:
        if logger:
            logger.debug(f"No tiling headers found in {host_dir}")
        return False

    any_injected = False
    for header_path in headers:
        tdef = parse_tiling_header(header_path)
        if not tdef or not tdef.fields:
            if logger:
                logger.debug(f"No tiling fields found in {header_path}")
            continue

        if logger:
            field_names = [f.name for f in tdef.fields]
            logger.info(f"Parsed {tdef.class_name}: {len(tdef.fields)} fields ({', '.join(field_names[:5])}...)")

        # Find cpp files with SaveToBuffer in op_host/
        cpp_files = glob.glob(os.path.join(host_dir, "*.cpp"))
        for cpp_path in cpp_files:
            with open(cpp_path, "r", encoding="utf-8") as f:
                if "SaveToBuffer" in f.read():
                    if _inject_into_cpp(cpp_path, tdef.fields, logger):
                        any_injected = True

    return any_injected
