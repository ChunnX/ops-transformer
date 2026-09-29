"""Conservative, read-only Ascend C flag candidate inventory; NOT a C++ verifier."""
import argparse
import ast
import itertools
import json
from pathlib import Path
import re

CALL = re.compile(r"\b(CrossCoreSetFlag|CrossCoreWaitFlag|NotifyEvent|WaitEvent)\s*(?:<([^;{}]*?)>)?\s*\(([^;{}]*?)\)", re.S)
DECL = re.compile(r"\b(?:constexpr|const)\s+[\w:]+\s+(\w+)\s*(?:\[[^\]]*\])?\s*=\s*([^;]+);", re.S)


def strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)


def values(expression, symbols):
    expression = re.sub(r"\b(\w+)\s*\[[^\]]+\]", r"\1", expression.strip())
    expression = re.sub(r"(0x[0-9a-fA-F]+|\b[0-9]+)[uUlL]+\b", r"\1", expression)
    if expression.startswith("{") and expression.endswith("}"):
        parts = [values(part, symbols) for part in expression[1:-1].split(",") if part.strip()]
        return None if any(p is None for p in parts) else set().union(*parts)

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return {node.value}
        if isinstance(node, ast.Name):
            return symbols.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = visit(node.operand)
            return None if operand is None else {(-v if isinstance(node.op, ast.USub) else v) for v in operand}
        if isinstance(node, ast.BinOp):
            lhs, rhs = visit(node.left), visit(node.right)
            if lhs is None or rhs is None or len(lhs) * len(rhs) > 256:
                return None
            functions = {ast.Add: lambda a, b: a+b, ast.Sub: lambda a, b: a-b,
                         ast.BitOr: lambda a, b: a|b, ast.BitAnd: lambda a, b: a&b,
                         ast.BitXor: lambda a, b: a^b}
            fn = functions.get(type(node.op))
            return None if fn is None else {fn(a, b) for a, b in itertools.product(lhs, rhs)}
        return None
    try:
        return visit(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, TypeError):
        return None


def scan(sources, overrides=None):
    symbols = {key: {value} for key, value in (overrides or {}).items()}
    locked = set(symbols)
    clean = {name: strip_comments(text) for name, text in sources.items()}
    declarations = []
    for text in clean.values():
        declarations += DECL.findall(text)
        declarations += re.findall(r"(?m)^\s*#define\s+(\w+)[ \t]+([^\n]+)", text)
    # A monotone union preserves uncertainty from inactive branches/redefinitions.
    for _ in range(len(declarations) + 1):
        changed = False
        for name, expression in declarations:
            if name in locked:
                continue
            result = values(expression, symbols)
            if result is not None:
                result |= symbols.get(name, set())
                if len(result) <= 256 and result != symbols.get(name):
                    symbols[name] = result
                    changed = True
        if not changed:
            break
    calls, warnings = [], []
    for name, text in clean.items():
        for match in CALL.finditer(text):
            api, template, expression = match.groups()
            args = [] if template is None else [arg.strip() for arg in template.split(",")]
            # CrossCoreWaitFlag's documented 910B/CANN 9.1 default is 0.
            # NotifyEvent/WaitEvent mode is left unknown: version/architecture sensitive.
            mode = values(args[0], symbols) if args and api.startswith("CrossCore") else ({0} if api == "CrossCoreWaitFlag" else None)
            flags = values(expression, symbols)
            calls.append(dict(file=name, line=text.count("\n", 0, match.start()) + 1,
                              api=api, template=template, expression=expression.strip(),
                              mode=sorted(mode) if mode is not None else None,
                              flags=sorted(flags) if flags is not None else None))
        if re.search(r"\b(SyncAll|clearWorkspace|Matmul)\b|matmul_intf\.h", text):
            warnings.append(f"{name}: inspect implicit/library synchronization in the actual ELF")
    producers = {}
    for index, call in enumerate(calls):
        if call["api"] != "CrossCoreSetFlag" or call["mode"] is None or call["flags"] is None:
            continue
        for flag in call["flags"]:
            for mode in call["mode"]:
                producers.setdefault(flag, {}).setdefault(mode, []).append(index)
    candidates = [dict(flag=flag, modes=sorted(modes), call_indices=sorted({i for group in modes.values() for i in group}))
                  for flag, modes in sorted(producers.items()) if len(modes) > 1]
    unresolved = [i for i, call in enumerate(calls) if call["flags"] is None or call["mode"] is None]
    return dict(scope="lexical candidates only; branches/namespaces are over-approximated; no safety proof",
                calls=calls, candidates=candidates, unresolved=unresolved, warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("-D", "--define", action="append", default=[])
    parser.add_argument("--fail-on-candidate", action="store_true")
    args = parser.parse_args()
    overrides = {}
    for item in args.define:
        try:
            key, value = item.split("=", 1)
            if not re.fullmatch(r"[A-Za-z_]\w*", key):
                raise ValueError()
            overrides[key] = int(value, 0)
        except ValueError:
            parser.error("-D expects NAME=integer")
    paths = set()
    for path in args.paths:
        if not path.exists():
            parser.error(f"missing input: {path}")
        if path.is_dir():
            paths.update(p for p in path.rglob("*") if p.is_file() and p.suffix in (".h", ".hpp", ".cpp", ".cc"))
        else:
            paths.add(path)
    if not paths:
        parser.error("no source files")
    result = scan({str(path): path.read_text() for path in sorted(paths)}, overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_candidate and result["candidates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
