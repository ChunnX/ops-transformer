#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build an entry-scenario guard plan from an aclnn interface header."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROTO_RE = re.compile(
    r"(?:__attribute__\s*\(\(.*?\)\)\s*)?aclnnStatus\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*;",
    re.S,
)


@dataclass
class Parameter:
    type_name: str
    name: str
    role: str
    optional: bool


@dataclass
class ApiFunction:
    name: str
    params: list[Parameter]


@dataclass
class Axis:
    name: str
    values: list[str]
    source: str


@dataclass
class Rule:
    selector: dict[str, list[str]]
    action: str
    reason: str


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*?$", "", text, flags=re.M)


def split_params(params: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in params:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                result.append(item)
            current = []
            continue
        current.append(ch)
    item = "".join(current).strip()
    if item and item != "void":
        result.append(item)
    return result


def parse_param(raw: str) -> Parameter:
    raw = raw.split("=", 1)[0].strip()
    match = re.match(r"(?P<type>.*?)(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?$", raw)
    if not match:
        return Parameter(raw, "unknown", "input", False)
    type_name = " ".join(match.group("type").strip().split())
    name = match.group("name")
    optional = "Optional" in name or name.endswith("Optional")
    role = "input"
    if name in {"out", "y", "workspaceSize", "executor"}:
        role = "output"
    elif "const" not in type_name and ("aclTensorList" in type_name or name.endswith("OutOptional")):
        role = "output"
    return Parameter(type_name, name, role, optional)


def parse_header(path: Path) -> list[ApiFunction]:
    text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    funcs: list[ApiFunction] = []
    for name, params_text in PROTO_RE.findall(text):
        params = [parse_param(item) for item in split_params(params_text)]
        funcs.append(ApiFunction(name, params))
    return funcs


def values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


SCALAR_VALUE_HINTS = {
    "splitItem": ["0", "1", "2", "3"],
    "groupType": ["-1", "0", "1", "2"],
    "groupListType": ["0", "1", "2"],
    "actType": ["0:none", "1", "2", "3", "4", "5:invalid"],
    "innerPrecise": ["0", "1"],
    "sparseMode": ["0", "1", "2", "3"],
    "pseType": ["0", "1"],
    "layout": ["BSH", "BNSD", "SBH"],
    "inputLayout": ["BSH", "BNSD", "SBH"],
}


def is_tensor_param(param: Parameter) -> bool:
    return "aclTensor" in param.type_name and "aclTensorList" not in param.type_name


def is_tensor_list_param(param: Parameter) -> bool:
    return "aclTensorList" in param.type_name


def is_array_param(param: Parameter) -> bool:
    return "aclIntArray" in param.type_name or "aclFloatArray" in param.type_name or \
        "aclBoolArray" in param.type_name


def is_scalar_param(param: Parameter) -> bool:
    scalar_types = ("int64_t", "uint64_t", "int32_t", "uint32_t", "double", "float", "bool")
    return any(item in param.type_name for item in scalar_types) and "*" not in param.type_name


def suggest_axes(func: ApiFunction) -> list[Axis]:
    axes: list[Axis] = [Axis("platform", ["Ascend910B", "Ascend310P", "Ascend950"], "default")]
    tensor_inputs: list[str] = []
    for param in func.params:
        if param.name in {"workspaceSize", "executor"}:
            continue
        if is_tensor_list_param(param):
            values_for_size = ["nullptr", "0", "1", ">1"] if param.optional else ["1", ">1"]
            axes.append(Axis(f"{param.name}_list_size", values_for_size, "tensor-list"))
            axes.append(Axis(f"{param.name}_dtype", ["fp16", "bf16", "fp32", "int8", "int32", "other"],
                             "tensor-list"))
            axes.append(Axis(f"{param.name}_format", ["ND", "NZ", "other"], "tensor-list"))
            tensor_inputs.append(param.name)
        elif is_tensor_param(param):
            if param.optional:
                axes.append(Axis(f"{param.name}_present", ["nullptr", "present"], "optional-tensor"))
            axes.append(Axis(f"{param.name}_rank", ["0", "1", "2", ">2"], "tensor"))
            axes.append(Axis(f"{param.name}_dtype", ["fp16", "bf16", "fp32", "int8", "int32", "other"], "tensor"))
            axes.append(Axis(f"{param.name}_format", ["ND", "NZ", "other"], "tensor"))
            tensor_inputs.append(param.name)
        elif is_array_param(param):
            values_for_size = ["nullptr", "0", "1", ">1"] if param.optional or "*" in param.type_name else ["0", "1", ">1"]
            axes.append(Axis(f"{param.name}_size", values_for_size, "array"))
        elif is_scalar_param(param):
            scalar_values = SCALAR_VALUE_HINTS.get(param.name)
            if scalar_values is None:
                scalar_values = ["valid", "boundary", "invalid"]
            axes.append(Axis(param.name, scalar_values, "scalar"))
        elif "*" in param.type_name and "char" in param.type_name:
            axes.append(Axis(f"{param.name}_present", ["nullptr", "present"], "pointer"))
    if len(tensor_inputs) >= 2:
        axes.append(Axis("dtype_pattern", ["same_dtype", "mixed_dtype", "quant", "unknown"], "cross-tensor"))
    return dedupe_axes(axes)


def dedupe_axes(axes: Iterable[Axis]) -> list[Axis]:
    result: list[Axis] = []
    seen: set[str] = set()
    for axis in axes:
        if axis.name in seen:
            continue
        seen.add(axis.name)
        result.append(axis)
    return result


def prompt(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{message}{suffix}: ").strip()
    return default if answer == "" and default is not None else answer


def choose_function(funcs: list[ApiFunction], requested: str | None, non_interactive: bool) -> ApiFunction:
    if not funcs:
        raise SystemExit("No aclnnStatus function prototype was found.")
    if requested:
        for func in funcs:
            if func.name == requested:
                return func
        raise SystemExit(f"Function {requested!r} was not found in the header.")
    if non_interactive or len(funcs) == 1:
        return funcs[0]
    print("Functions:")
    for idx, func in enumerate(funcs, start=1):
        print(f"  {idx}. {func.name}")
    while True:
        selected = prompt("Select function number", "1")
        if selected.isdigit() and 1 <= int(selected) <= len(funcs):
            return funcs[int(selected) - 1]


def ask_axes(suggested: list[Axis], non_interactive: bool) -> list[Axis]:
    if non_interactive:
        return suggested
    selected: list[Axis] = []
    print("\nSuggested scenario axes:")
    for axis in suggested:
        default_values = ",".join(axis.values)
        use_axis = prompt(f"Use axis {axis.name} from {axis.source}? y/n", "y").lower()
        if use_axis.startswith("n"):
            continue
        edited = prompt(f"Values for {axis.name}", default_values)
        selected.append(Axis(axis.name, values(edited), axis.source))
    while True:
        name = prompt("Add custom axis name, blank to finish", "")
        if not name:
            break
        selected.append(Axis(name, values(prompt(f"Values for {name}")), "custom"))
    return selected


def parse_selector(text: str) -> dict[str, list[str]]:
    selector: dict[str, list[str]] = {}
    for item in values(text):
        if "=" not in item:
            raise ValueError(f"Selector item {item!r} must be key=value.")
        key, raw_value = item.split("=", 1)
        selector[key.strip()] = [part.strip() for part in raw_value.split("|") if part.strip()]
    return selector


def format_selector(selector: dict[str, list[str]]) -> str:
    if not selector:
        return "*"
    return ", ".join(f"{key}={'|'.join(vals)}" for key, vals in selector.items())


def ask_rules(non_interactive: bool) -> tuple[list[Rule], str]:
    if non_interactive:
        return [], "allow"
    print("\nAdd interception rules. Example: platform=Ascend310P,input_dtype=int8")
    rules: list[Rule] = []
    while True:
        raw_selector = prompt("Rule selector, blank to finish", "")
        if not raw_selector:
            break
        try:
            selector = parse_selector(raw_selector)
        except ValueError as err:
            print(err)
            continue
        action = prompt("Action allow/block", "block").lower()
        if action not in {"allow", "block"}:
            print("Action must be allow or block.")
            continue
        reason = prompt("Reason", "unsupported scenario")
        rules.append(Rule(selector, action, reason))
    default_action = prompt("Default action for unmatched combinations allow/block", "allow").lower()
    if default_action not in {"allow", "block"}:
        default_action = "allow"
    return rules, default_action


def combo_count(axes: list[Axis]) -> int:
    if not axes:
        return 1
    return math.prod(max(1, len(axis.values)) for axis in axes)


def iter_combinations(axes: list[Axis]) -> Iterable[dict[str, str]]:
    if not axes:
        yield {}
        return
    names = [axis.name for axis in axes]
    for product in itertools.product(*(axis.values for axis in axes)):
        yield dict(zip(names, product))


def rule_matches(combo: dict[str, str], selector: dict[str, list[str]]) -> bool:
    for key, expected_values in selector.items():
        value = combo.get(key)
        if value is None:
            return False
        normalized_value = value.split(":", 1)[0]
        normalized_expected = [item.split(":", 1)[0] for item in expected_values]
        if "*" not in expected_values and value not in expected_values and normalized_value not in normalized_expected:
            return False
    return True


def decide(combo: dict[str, str], rules: list[Rule], default_action: str) -> tuple[str, str]:
    for rule in rules:
        if rule_matches(combo, rule.selector):
            return rule.action, rule.reason
    return default_action, "default"


def build_matrix(axes: list[Axis], rules: list[Rule], default_action: str, limit: int) -> tuple[list[dict[str, object]], bool]:
    count = combo_count(axes)
    rows: list[dict[str, object]] = []
    for idx, combo in enumerate(iter_combinations(axes)):
        if idx >= limit:
            return rows, count > limit
        action, reason = decide(combo, rules, default_action)
        rows.append({"conditions": combo, "action": action, "reason": reason})
    return rows, False


def render_markdown(func: ApiFunction, header: Path, axes: list[Axis], rules: list[Rule], default_action: str,
                    matrix: list[dict[str, object]], truncated: bool) -> str:
    lines = [
        f"# {func.name} scenario guard plan",
        "",
        f"- Header: `{header}`",
        f"- Default action: `{default_action}`",
        f"- Combination count: `{combo_count(axes)}`",
        "",
        "## Interface parameters",
        "",
        "| Name | Type | Role | Optional |",
        "| --- | --- | --- | --- |",
    ]
    for param in func.params:
        lines.append(f"| `{param.name}` | `{param.type_name}` | `{param.role}` | `{str(param.optional).lower()}` |")
    lines.extend([
        "",
        "## Stable entry validation order",
        "",
        "1. Required pointer and tensor-list non-null checks.",
        "2. DFX phase-1 logging after required inputs are known to be safe.",
        "3. Platform and interface-version support checks.",
        "4. Scalar and enum-like attribute checks.",
        "5. Optional input presence, rank, dtype, format, and array-size checks.",
        "6. Interface-specific scenario intercepts recorded in the decision matrix.",
        "7. Data representation normalization, if this interface has such conversions.",
        "8. Deep dtype, format, shape, optional-parameter, and output checks.",
        "",
        "## Scenario axes",
        "",
        "| Axis | Values | Source |",
        "| --- | --- | --- |",
    ])
    for axis in axes:
        lines.append(f"| `{axis.name}` | {', '.join(f'`{value}`' for value in axis.values)} | {axis.source} |")
    lines.extend(["", "## Interception rules", "", "| Selector | Action | Reason |", "| --- | --- | --- |"])
    if rules:
        for rule in rules:
            lines.append(f"| `{format_selector(rule.selector)}` | `{rule.action}` | {rule.reason} |")
    else:
        lines.append("| `*` | default | No explicit rules were recorded. |")
    lines.extend(["", "## Decision matrix", "", "| Conditions | Action | Reason |", "| --- | --- | --- |"])
    for row in matrix:
        condition = ", ".join(f"{key}={value}" for key, value in row["conditions"].items()) or "*"
        lines.append(f"| `{condition}` | `{row['action']}` | {row['reason']} |")
    if truncated:
        lines.append("| ... | ... | Matrix truncated by max-combinations. |")
    lines.append("")
    return "\n".join(lines)


def cpp_identifier(name: str) -> str:
    result = re.sub(r"\W", "_", name)
    if not result or result[0].isdigit():
        result = f"field_{result}"
    return result


def selector_value_literal(value: str) -> str:
    token = value.split(":", 1)[0]
    if re.fullmatch(r"-?\d+", token):
        return token
    if token in {"true", "false"}:
        return token
    escaped = token.replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{escaped}\""


def selector_field_type(values_for_field: list[str]) -> str:
    tokens = [value.split(":", 1)[0] for value in values_for_field]
    if tokens and all(re.fullmatch(r"-?\d+", token) for token in tokens):
        return "int64_t"
    if tokens and all(token in {"true", "false"} for token in tokens):
        return "bool"
    return "std::string"


def render_skeleton(func: ApiFunction, axes: list[Axis], rules: list[Rule]) -> str:
    guard_name = f"Check{func.name}EntryScenario"
    selector_values: dict[str, list[str]] = {}
    for rule in rules:
        for key, vals in rule.selector.items():
            selector_values.setdefault(key, [])
            selector_values[key].extend(vals)
    if not selector_values:
        for axis in axes[:8]:
            selector_values[axis.name] = axis.values

    lines = [
        f"// Skeleton generated by api_scenario_guard_wizard.py for {func.name}.",
        "// Replace EntryScenario fields with values derived from real aclnn parameters.",
        "#include <string>",
        "",
        "struct EntryScenario {",
    ]
    for key, vals in selector_values.items():
        field_type = selector_field_type(vals)
        default_value = "0" if field_type == "int64_t" else ("false" if field_type == "bool" else "\"\"")
        lines.append(f"  {field_type} {cpp_identifier(key)} = {default_value};")
    lines.extend([
        "};",
        "",
        f"static aclnnStatus {guard_name}(const EntryScenario &scenario)",
        "{",
    ])
    for rule in rules:
        checks = []
        for key, vals in rule.selector.items():
            field = f"scenario.{cpp_identifier(key)}"
            checks.append(" || ".join(f"{field} == {selector_value_literal(value)}" for value in vals))
        if checks and rule.action == "block":
            lines.append(f"  if ({' && '.join(f'({check})' for check in checks)}) {{")
            reason = rule.reason.replace("\\", "\\\\").replace("\"", "\\\"")
            lines.append(f"    OP_LOGE(ACLNN_ERR_PARAM_INVALID, \"{reason}\");")
            lines.append("    return ACLNN_ERR_PARAM_INVALID;")
            lines.append("  }")
    lines.extend(["  return ACLNN_SUCCESS;", "}", ""])
    return "\n".join(lines)


def write_outputs(out_dir: Path, header: Path, func: ApiFunction, axes: list[Axis], rules: list[Rule],
                  default_action: str, max_combinations: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix, truncated = build_matrix(axes, rules, default_action, max_combinations)
    payload = {
        "header": str(header),
        "function": {
            "name": func.name,
            "parameters": [param.__dict__ for param in func.params],
        },
        "axes": [axis.__dict__ for axis in axes],
        "rules": [{"selector": rule.selector, "action": rule.action, "reason": rule.reason} for rule in rules],
        "default_action": default_action,
        "combination_count": combo_count(axes),
        "matrix_truncated": truncated,
        "matrix": matrix,
    }
    stem = func.name
    (out_dir / f"{stem}_scenario_guard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / f"{stem}_scenario_guard.md").write_text(
        render_markdown(func, header, axes, rules, default_action, matrix, truncated), encoding="utf-8"
    )
    (out_dir / f"{stem}_scenario_guard_skeleton.cpp").write_text(render_skeleton(func, axes, rules), encoding="utf-8")
    print(f"Wrote scenario guard plan to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path, help="Path to an aclnn interface .h file.")
    parser.add_argument("--function", help="Function name to inspect when the header has multiple aclnnStatus APIs.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/generated_scenario_guards"),
                        help="Directory for generated JSON, Markdown, and C++ skeleton files.")
    parser.add_argument("--max-combinations", type=int, default=500,
                        help="Maximum decision-matrix rows to materialize.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Use suggested axes, no explicit interception rules, and default allow.")
    args = parser.parse_args()

    funcs = parse_header(args.header)
    func = choose_function(funcs, args.function, args.non_interactive)
    print(f"Selected function: {func.name}")
    axes = ask_axes(suggest_axes(func), args.non_interactive)
    rules, default_action = ask_rules(args.non_interactive)
    write_outputs(args.out_dir, args.header, func, axes, rules, default_action, args.max_combinations)


if __name__ == "__main__":
    main()
