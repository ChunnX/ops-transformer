#!/usr/bin/env python3
"""Bulk-transform non-standard OP_LOGE to CANN standard ErrMessage templates."""

import argparse
import re
from pathlib import Path

DEFAULT_OP_API_NAME = "FusedInferAttentionScore"
DEFAULT_OP_HOST_FALLBACKS = ('"FusedInferAttentionScore"', '"FIA"')

STANDARD_MARKERS = ("The op [%s]", "In op [")

TEMPLATES = {
    "bad_param": '"The op [%s] received bad params, the reason is: [%s]"',
    "missing_param": '"The op [%s] missing [%s] parameter"',
    "missing_attr": '"The op [%s] missing [%s] attribute"',
    "shape_mismatch": '"In op [%s], the tensor shapes of [%s..] are mismatched, the reason is: [%s]"',
    "dtype_mismatch": '"In op [%s], the data types of [%s..] are mismatched, the reason is: [%s]"',
    "not_supported_feat": '"In op [%s], [%s] is not supported, got [%s]"',
    "dtype_not_supported": '"In op [%s], the data type of [%s] is not supported, got [%s]"',
    "shape_not_supported": '"In op [%s], the shape of [%s] is not supported, got [%s]. Constraint: [%s]"',
}


def extract_op_loge_calls(text: str):
    """Yield (start, end, full_match) for each OP_LOGE(...)."""
    i = 0
    while True:
        idx = text.find("OP_LOGE(", i)
        if idx == -1:
            break
        depth = 0
        j = idx + len("OP_LOGE")
        while j < len(text):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    yield idx, j + 1, text[idx : j + 1]
                    i = j + 1
                    break
            j += 1
        else:
            break


def split_first_arg(rest: str):
    """Split first comma-separated arg respecting parens/quotes."""
    depth = 0
    in_str = False
    esc = False
    for i, c in enumerate(rest):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return rest[:i].strip(), rest[i + 1 :].strip()
    return rest.strip(), ""


def parse_format_and_args(remainder: str):
    """Parse concatenated C string literals as format and remaining args."""
    fmt_parts = []
    pos = 0
    s = remainder.lstrip()
    while pos < len(s) and s[pos] == '"':
        end = pos + 1
        esc = False
        while end < len(s):
            c = s[end]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                break
            end += 1
        fmt_parts.append(s[pos + 1 : end])
        pos = end + 1
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
    fmt = "".join(fmt_parts)
    tail = s[pos:].lstrip()
    if tail.startswith(","):
        tail = tail[1:].lstrip()
        args = split_args(tail) if tail else []
    else:
        args = []
    return fmt, args


def parse_call(call: str):
    inner = call[len("OP_LOGE(") : -1]
    first, remainder = split_first_arg(inner)
    if not remainder:
        return first, "", []
    fmt, args = parse_format_and_args(remainder)
    return first, fmt, args


def split_args(s: str):
    """Split comma-separated args."""
    parts = []
    depth = 0
    in_str = False
    esc = False
    start = 0
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    if s[start:].strip():
        parts.append(s[start:].strip())
    return parts


def infer_param_name(msg: str) -> str:
    msg_l = msg.lower()
    m = re.search(r"(?:tensor|shape of tensor|desc of tensor|desc of)\s+(\w+)", msg_l)
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s+is null", msg_l)
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s+input", msg_l)
    if m:
        return m.group(1)
    return "input"


def infer_attr_name(msg: str) -> str:
    m = re.search(r"attr\s+(\w+)", msg, re.I)
    if m:
        name = m.group(1)
        # camelCase to snake_case-ish for CANN names
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return "attribute"


def classify(msg: str, args):
    msg_l = msg.lower()
    if "attr" in msg_l and ("nullptr" in msg_l or "is null" in msg_l):
        return "missing_attr", infer_attr_name(msg)
    if ("nullptr" in msg_l or "is null" in msg_l) and (
        "tensor" in msg_l or "shape" in msg_l or "desc" in msg_l or "input" in msg_l
    ):
        return "missing_param", infer_param_name(msg)
    if ("datatype" in msg_l or "dtype" in msg_l or "data type" in msg_l) and (
        "mismatch" in msg_l or "should be equal" in msg_l or "must be equal" in msg_l
    ):
        return "dtype_mismatch", msg
    if ("datatype" in msg_l or "dtype" in msg_l) and (
        "not support" in msg_l or "only support" in msg_l
    ):
        return "dtype_not_supported", msg
    if "shape" in msg_l and (
        "mismatch" in msg_l or "does not match" in msg_l or "not equal" in msg_l
    ):
        return "shape_mismatch", msg
    if ("dim" in msg_l or "shape" in msg_l) and (
        "must be" in msg_l or "should be" in msg_l or "not support" in msg_l
    ):
        return "shape_not_supported", msg
    if "not support" in msg_l or "not yet support" in msg_l or "do not support" in msg_l:
        return "not_supported_feat", msg
    return "bad_param", msg


def is_inside_op_check_if(text: str, op_loge_start: int) -> bool:
    """Return True if OP_LOGE at start is an argument of OP_CHECK_IF."""
    prefix = text[:op_loge_start]
    idx = prefix.rfind("OP_CHECK_IF(")
    if idx == -1:
        return False
    depth = 0
    for c in prefix[idx:]:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return depth > 0


def transform_call(first: str, fmt: str, args, is_op_api: bool, inside_check: bool, op_api_name: str,
                   op_host_fallbacks: tuple) -> str | None:
    if not fmt or fmt.strip() == "":
        return None
    if any(m in fmt for m in STANDARD_MARKERS):
        return None

    kind, detail = classify(fmt, args)

    if is_op_api:
        op_name_arg = f'"{op_api_name}"'
    elif first in op_host_fallbacks:
        op_name_arg = f'"{op_api_name}"'
    else:
        op_name_arg = first

    if kind == "missing_attr":
        attr = detail if isinstance(detail, str) and detail != "attribute" else infer_attr_name(fmt)
        log = f"OP_LOGE({first}, {TEMPLATES['missing_attr']}, {op_name_arg}, \"{attr}\")"
        return wrap_stmt_expr(log, inside_check, False)

    if kind == "missing_param":
        param = detail if isinstance(detail, str) else infer_param_name(fmt)
        log = f"OP_LOGE({first}, {TEMPLATES['missing_param']}, {op_name_arg}, \"{param}\")"
        return wrap_stmt_expr(log, inside_check, False)

    reason_literal = fmt.replace('"', '\\"')

    if not args:
        if kind == "not_supported_feat":
            feat = "feature"
            got = reason_literal[:120]
            log = (
                f"OP_LOGE({first}, {TEMPLATES['not_supported_feat']}, "
                f"{op_name_arg}, \"{feat}\", \"{got}\")"
            )
        else:
            log = (
                f"OP_LOGE({first}, {TEMPLATES['bad_param']}, "
                f"{op_name_arg}, \"{reason_literal}\")"
            )
        return wrap_stmt_expr(log, inside_check, False)

    # Has format args -> use snprintf bad_param
    old_fmt = fmt.replace('"', '\\"')
    snprintf_args = ", ".join(args)
    fallback = reason_literal[:120].replace('"', '\\"')
    reason_buf = (
        f"(snprintf(reason, sizeof(reason), \"{old_fmt}\", {snprintf_args}) > 0) "
        f"? reason : \"{fallback}\""
    )
    log = f"OP_LOGE({first}, {TEMPLATES['bad_param']}, {op_name_arg}, {reason_buf})"
    return wrap_stmt_expr(log, inside_check, True)


def wrap_stmt_expr(log: str, inside_check: bool, need_reason: bool) -> str:
    if need_reason and inside_check:
        return f"({{ char reason[512] = {{0}}; {log}; }})"
    if need_reason and not inside_check:
        return log
    if inside_check:
        return f"({{ {log}; }})"
    return log


def needs_cstdio(text: str) -> bool:
    return "snprintf(" in text and "#include <cstdio>" not in text


def add_cstdio(text: str) -> str:
    if "#include <cstdio>" in text:
        return text
    m = re.search(r'#include\s+[<"][^>"]+[>"]', text)
    if m:
        insert_at = m.end()
        return text[:insert_at] + "\n#include <cstdio>" + text[insert_at:]
    return "#include <cstdio>\n" + text


def add_reason_decl(text: str) -> str:
    if "char reason[512]" in text:
        return text
    # Insert after opening brace of first function that uses snprintf after transform
    if "snprintf(reason" not in text:
        return text
    # Add at start of each function body that contains snprintf - simplified: one global per file at namespace
    if "namespace optiling" in text or "namespace l0op" in text:
        marker = "namespace optiling {\n"
        if marker in text and "char reason[512]" not in text.split("namespace optiling {", 1)[1][:500]:
            # Insert helper at file level inside namespace - actually per-function is safer
            pass
    return text


def process_file(path: Path, op_api_name: str, op_host_fallbacks: tuple) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original
    is_op_api = "/op_api/" in str(path) or (
        path.name.startswith("aclnn_") and "/op_api" in str(path.parent)
    )

    replacements = []
    for start, end, call in extract_op_loge_calls(text):
        first, fmt, args = parse_call(call)
        if not fmt:
            continue
        new_call = transform_call(first, fmt, args, is_op_api, is_inside_op_check_if(text, start),
                                  op_api_name, op_host_fallbacks)
        if new_call and new_call != call:
            replacements.append((start, end, new_call))

    if not replacements:
        return False

    # Apply from end to start
    for start, end, new_call in reversed(replacements):
        text = text[:start] + new_call + text[end:]

    if needs_cstdio(text):
        text = add_cstdio(text)

    if "snprintf(reason" in text:
        lines = text.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            if (
                "OP_LOGE(" in line
                and "snprintf(reason" in line
                and "({ char reason[512]" not in line
            ):
                window = "\n".join(lines[max(0, i - 40) : i + 1])
                if "char reason[512]" not in window:
                    indent = re.match(r"^(\s*)", line).group(1)
                    new_lines.append(f"{indent}char reason[512] = {{0}};")
            new_lines.append(line)
        text = "\n".join(new_lines)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Transform OP_LOGE to standard ErrMessage format")
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        help="Directories to scan for *.cpp",
    )
    parser.add_argument(
        "--op-api-name",
        default=DEFAULT_OP_API_NAME,
        help="PascalCase operator name for op_api layer format strings",
    )
    parser.add_argument(
        "--op-host-fallback",
        nargs="*",
        default=list(DEFAULT_OP_HOST_FALLBACKS),
        help='Hardcoded op_host first args to map to op-api-name, e.g. "IncreFlashAttention"',
    )
    args = parser.parse_args()
    op_host_fallbacks = tuple(args.op_host_fallback)
    if not args.dirs:
        args.dirs = [
            Path("/home/xxp/ops-transformer/attention/fused_infer_attention_score/op_host"),
            Path("/home/xxp/ops-transformer/attention/fused_infer_attention_score/op_api"),
        ]

    changed = []
    for op_dir in args.dirs:
        if not op_dir.exists():
            print(f"skip missing: {op_dir}")
            continue
        for cpp in sorted(op_dir.rglob("*.cpp")):
            if process_file(cpp, args.op_api_name, op_host_fallbacks):
                changed.append(str(cpp))
    print(f"Updated {len(changed)} files")
    for f in changed:
        print(f"  {f}")


if __name__ == "__main__":
    main()
