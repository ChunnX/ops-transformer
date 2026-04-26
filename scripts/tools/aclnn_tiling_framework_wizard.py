#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate an aclnn + tiling + kernel-dispatch framework from existing operator interface files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.api_scenario_guard_wizard import ApiFunction, Parameter, parse_header


@dataclass
class DefTensor:
    kind: str
    name: str
    param_type: str
    dtypes: list[str]
    formats: list[str]
    auto_contiguous: bool


@dataclass
class DefAttr:
    name: str
    attr_type: str
    value_type: str
    default_value: str


@dataclass
class OpDefInfo:
    op_type: str
    file_stem: str
    inputs: list[DefTensor]
    outputs: list[DefTensor]
    attrs: list[DefAttr]
    configs: list[str]


@dataclass
class TemplateCase:
    tiling_key: str
    dtype_guard: str
    invoke: str
    args: str


def snake_to_camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def camel_to_snake(name: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def op_file_stem(op_type: str, def_info: OpDefInfo | None = None) -> str:
    if def_info and def_info.file_stem:
        return def_info.file_stem.removesuffix("_def")
    return camel_to_snake(op_type)


def derive_op_type(func: ApiFunction, def_info: OpDefInfo | None) -> str:
    if def_info and def_info.op_type:
        return def_info.op_type
    name = func.name
    if name.startswith("aclnn"):
        name = name[len("aclnn"):]
    if name.endswith("GetWorkspaceSize"):
        name = name[: -len("GetWorkspaceSize")]
    return name


def parse_braced_values(text: str) -> list[str]:
    return [item.strip().replace("ge::", "") for item in text.split(",") if item.strip()]


def parse_tensor_block(kind: str, name: str, block: str) -> DefTensor:
    param_type = re.search(r"\.ParamType\((\w+)\)", block)
    dtypes = re.search(r"\.DataType(?:List)?\(\{([^}]*)\}\)", block, re.S)
    formats = re.search(r"\.Format(?:List)?\(\{([^}]*)\}\)", block, re.S)
    return DefTensor(
        kind=kind,
        name=name,
        param_type=param_type.group(1) if param_type else "UNKNOWN",
        dtypes=parse_braced_values(dtypes.group(1)) if dtypes else [],
        formats=parse_braced_values(formats.group(1)) if formats else [],
        auto_contiguous=".AutoContiguous()" in block,
    )


def parse_def_cpp(path: Path | None) -> OpDefInfo | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    op_add = re.search(r"OP_ADD\((\w+)\)", text)
    op_class = re.search(r"class\s+(\w+)\s*:\s*public\s+OpDef", text)
    op_type = op_add.group(1) if op_add else (op_class.group(1) if op_class else path.stem)
    tensors: list[DefTensor] = []
    for kind in ("Input", "Output"):
        pattern = re.compile(rf"this->{kind}\(\"([^\"]+)\"\)(.*?);", re.S)
        for name, block in pattern.findall(text):
            tensors.append(parse_tensor_block(kind.lower(), name, block))
    attrs: list[DefAttr] = []
    attr_pattern = re.compile(r'this->Attr\("([^"]+)"\)\s*\.AttrType\((\w+)\)\.(\w+)\((.*?)\);', re.S)
    for name, attr_type, value_type, default_value in attr_pattern.findall(text):
        attrs.append(DefAttr(name, attr_type, value_type, " ".join(default_value.split())))
    configs = re.findall(r'(?:->|\.)AICore\(\)\.AddConfig\("([^"]+)"\)', text)
    return OpDefInfo(
        op_type=op_type,
        file_stem=path.stem,
        inputs=[tensor for tensor in tensors if tensor.kind == "input"],
        outputs=[tensor for tensor in tensors if tensor.kind == "output"],
        attrs=attrs,
        configs=configs,
    )


def parse_kernel_templates(path: Path | None) -> list[TemplateCase]:
    if path is None:
        return []
    cases: list[TemplateCase] = []
    dtype_guard = "any"
    tiling_key = "any"
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        dtype_match = re.search(r"ORIG_DTYPE_([A-Za-z0-9_]+)\s*==\s*(DT_[A-Za-z0-9_]+)", line)
        if dtype_match:
            dtype_guard = f"ORIG_DTYPE_{dtype_match.group(1)}=={dtype_match.group(2)}"
        key_match = re.search(r"TILING_KEY_IS\(([^)]+)\)", line)
        if key_match:
            tiling_key = key_match.group(1).strip()
        invoke_match = re.search(r"(INVOKE_[A-Za-z0-9_]+)\((.*)\);", line)
        if invoke_match:
            cases.append(TemplateCase(tiling_key, dtype_guard, invoke_match.group(1), invoke_match.group(2).strip()))
            continue
        direct_template_match = re.search(r"([A-Za-z_]\w*)<([^>;]+)>\s+\w+\b", line)
        if direct_template_match and tiling_key != "any":
            template_name = direct_template_match.group(1)
            template_args = ", ".join(part.strip() for part in direct_template_match.group(2).split(","))
            direct_dtype_guard = dtype_guard
            if direct_dtype_guard == "any":
                direct_dtype_guard = f"template_args={template_args}"
            cases.append(TemplateCase(tiling_key, direct_dtype_guard, "DIRECT_TEMPLATE", f"{template_name}, {template_args}"))
    return cases


def parse_manual_template_categories(items: list[str]) -> list[TemplateCase]:
    cases: list[TemplateCase] = []
    for item in items:
        fields: dict[str, str] = {}
        for part in item.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        cases.append(TemplateCase(
            tiling_key=fields.get("tiling_key", fields.get("key", "TODO_KEY")),
            dtype_guard=fields.get("dtype_guard", fields.get("dtype", "any")),
            invoke=fields.get("invoke", "INVOKE_OP_IMPL"),
            args=fields.get("args", "TemplateClass, TODO_ARGS"),
        ))
    return cases


def cpp_type_for_param(param: Parameter) -> str:
    return f"{param.type_name} {param.name}"


def function_signature(func: ApiFunction) -> str:
    params = ",\n    ".join(cpp_type_for_param(param) for param in func.params)
    return f"aclnnStatus {func.name}(\n    {params})"


def required_tensor_params(func: ApiFunction, def_info: OpDefInfo | None) -> list[Parameter]:
    optional_names = set()
    if def_info:
        optional_names = {tensor.name for tensor in def_info.inputs + def_info.outputs if tensor.param_type == "OPTIONAL"}
    result = []
    for param in func.params:
        if "aclTensor" not in param.type_name:
            continue
        if param.name in {"workspaceSize", "executor"}:
            continue
        if param.optional or param.name in optional_names:
            continue
        result.append(param)
    return result


def attr_cpp_type(attr: DefAttr) -> str:
    mapping = {"Int": "int64_t", "Bool": "bool", "Float": "float", "Double": "double", "String": "const char *"}
    return mapping.get(attr.value_type, "auto")


def render_framework_md(func: ApiFunction, def_info: OpDefInfo | None, cases: list[TemplateCase],
                        aclnn_h: Path, def_cpp: Path | None, kernel_cpp: Path | None) -> str:
    op_type = derive_op_type(func, def_info)
    lines = [
        f"# {op_type} aclnn and tiling framework",
        "",
        "## Inputs",
        "",
        f"- aclnn header: `{aclnn_h}`",
        f"- op def: `{def_cpp}`" if def_cpp else "- op def: not provided",
        f"- kernel template source: `{kernel_cpp}`" if kernel_cpp else "- kernel template source: not provided",
        "",
        "## aclnn Entry Order",
        "",
        "1. Platform check: chip/SOC support must be decided before expensive checks.",
        "2. Data-type and format check by parameter category: aclTensorList, aclTensor, aclIntArray/aclFloatArray/aclBoolArray, scalar int/float/double/bool, and char/string parameters.",
        "3. Required input check: nullptr, empty tensor-list, output pointer, workspaceSize, executor.",
        "4. Optional input check: nullptr policy, presence controlled by flags/attrs, optional dtype/rank/shape when present.",
        "5. Combination-scene check: cross-parameter constraints that choose or block a scenario.",
        "6. Empty tensor and normalization handling: create executor, contiguous/cast/view conversion only after validation.",
        "7. L0 op call and ViewCopy to user outputs.",
        "",
        "## aclnn to Tiling Contract",
        "",
        "After aclnn succeeds, tiling should route valid cases to a template. Tiling may still check hardware limits, UB/workspace feasibility, and internal tiling-library return values. Kernel dispatch should trust tilingKey and compile-time dtype macros, then select the matching template instance.",
        "",
    ]
    if def_info:
        lines.extend(["## OpDef Summary", "", "| Kind | Name | ParamType | DTypes | Formats | AutoContiguous |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for tensor in def_info.inputs + def_info.outputs:
            lines.append(
                f"| {tensor.kind} | `{tensor.name}` | `{tensor.param_type}` | "
                f"{', '.join(tensor.dtypes) or '-'} | {', '.join(tensor.formats) or '-'} | "
                f"{str(tensor.auto_contiguous).lower()} |"
            )
        lines.extend(["", "## Attr Summary", "", "| Name | AttrType | ValueType | Default |", "| --- | --- | --- | --- |"])
        for attr in def_info.attrs:
            lines.append(f"| `{attr.name}` | `{attr.attr_type}` | `{attr.value_type}` | `{attr.default_value}` |")
        lines.extend(["", f"Supported configs: {', '.join(def_info.configs) or '-'}", ""])
    lines.extend(["## Template Categories", "", "| TilingKey | DType Guard | Invoke | Args |", "| --- | --- | --- | --- |"])
    if cases:
        for case in cases:
            lines.append(f"| `{case.tiling_key}` | `{case.dtype_guard}` | `{case.invoke}` | `{case.args}` |")
    else:
        lines.append("| `TODO_KEY` | `TODO_DTYPE_GUARD` | `INVOKE_OP_IMPL` | `TemplateClass, TODO_ARGS` |")
    lines.extend(["", "## Generated Files", "", "- `*_aclnn_skeleton.cpp`: entry validation and executor framework.",
                  "- `*_tiling_skeleton.cpp`: tiling parse/init/key/data framework.",
                  "- `*_kernel_dispatch_skeleton.cpp`: kernel tilingKey and template-dispatch framework.", ""])
    return "\n".join(lines)


def render_aclnn_skeleton(func: ApiFunction, def_info: OpDefInfo | None) -> str:
    op_type = derive_op_type(func, def_info)
    required = required_tensor_params(func, def_info)
    required_names = ", ".join(param.name for param in required) or "/* required tensors */"
    return f"""// Generated framework for {func.name}. Fill TODO checks with operator-specific constraints.
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

namespace {{
static aclnnStatus CheckPlatform()
{{
    // TODO: gate SOC/chip families from op_def AddConfig and product requirements.
    return ACLNN_SUCCESS;
}}

static aclnnStatus CheckDataTypesAndFormats()
{{
    // TODO: check aclTensorList, aclTensor, acl*Array, scalar, bool, float/double, and char/string parameters.
    return ACLNN_SUCCESS;
}}

static aclnnStatus CheckRequiredInputs()
{{
    // TODO: check required inputs: {required_names}.
    return ACLNN_SUCCESS;
}}

static aclnnStatus CheckOptionalInputs()
{{
    // TODO: check optional inputs and flag-controlled presence constraints.
    return ACLNN_SUCCESS;
}}

static aclnnStatus CheckScenarioCombinations()
{{
    // TODO: cross-parameter allow/block matrix; keep first-error order stable.
    return ACLNN_SUCCESS;
}}

static aclnnStatus CheckParams()
{{
    CHECK_RET(CheckPlatform() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDataTypesAndFormats() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRequiredInputs() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOptionalInputs() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScenarioCombinations() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}}
}} // namespace

{function_signature(func)}
{{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1({op_type}, DFX_IN(/* TODO inputs and attrs */), DFX_OUT(/* TODO outputs */));

    CHECK_RET(CheckParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // TODO: empty tensor fast path.
    // TODO: make required tensors contiguous/cast/format-normalized after validation.
    // TODO: call l0op::{op_type}(...) and ViewCopy results to outputs.

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}}
"""


def render_tiling_skeleton(op_type: str, def_info: OpDefInfo | None, cases: list[TemplateCase]) -> str:
    inputs = def_info.inputs if def_info else []
    attrs = def_info.attrs if def_info else []
    input_consts = "\n".join(
        f"constexpr int32_t {tensor.name.upper()}_INPUT_INDEX = {idx};" for idx, tensor in enumerate(inputs)
    )
    attr_consts = "\n".join(
        f"constexpr int32_t ATTR_{attr.name.upper()}_INDEX = {idx};" for idx, attr in enumerate(attrs)
    )
    attr_members = "\n".join(
        f"    {attr_cpp_type(attr)} {attr.name}_ = {{}};" for attr in attrs
    )
    attr_reads = "\n".join(
        f"    {attr.name}_ = *(attrs->GetAttrPointer<{attr_cpp_type(attr)}>(ATTR_{attr.name.upper()}_INDEX));"
        for attr in attrs if attr_cpp_type(attr) != "auto"
    )
    key_comment = "\n".join(
        f"    // key {case.tiling_key}: {case.dtype_guard}, {case.args}" for case in cases
    ) or "    // TODO: map scenario fields to tilingKey."
    return f"""// Generated tiling framework for {op_type}. Fill TODO resource and shape logic.
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "{op_file_stem(op_type, def_info)}_tiling.h"

namespace {{
{input_consts}
{attr_consts}
}} // namespace

namespace optiling {{
struct {op_type}CompileInfo {{
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
}};

class {op_type}Tiling {{
public:
    explicit {op_type}Tiling(gert::TilingContext *context) : context_(context) {{}}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();

private:
    void GetTilingKey();
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    uint32_t usedCoreNum_ = 1;
{attr_members}
    {op_type}TilingData tilingData_;
}};

ge::graphStatus {op_type}Tiling::Init()
{{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
{attr_reads}
    auto compileInfo = static_cast<const {op_type}CompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    // TODO: read input/output shapes, validate tiling-only hardware limits, split work.
    GetTilingKey();
    return ge::GRAPH_SUCCESS;
}}

void {op_type}Tiling::GetTilingKey()
{{
    tilingKey_ = 0;
{key_comment}
}}

ge::graphStatus {op_type}Tiling::SetKernelTiling()
{{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    // TODO: fill tilingData_ fields and workspace sizes.
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}}

static ge::graphStatus Tiling{op_type}(gert::TilingContext *context)
{{
    OP_CHECK_IF(context == nullptr, return ge::GRAPH_FAILED);
    {op_type}Tiling tiling(context);
    OP_CHECK_IF(tiling.Init() != ge::GRAPH_SUCCESS, return ge::GRAPH_FAILED);
    return tiling.SetKernelTiling();
}}

static ge::graphStatus TilingPrepare4{op_type}(gert::TilingParseContext *context)
{{
    auto compileInfo = context->GetCompiledInfo<{op_type}CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}}

IMPL_OP_OPTILING({op_type})
    .Tiling(Tiling{op_type})
    .TilingParse<{op_type}CompileInfo>(TilingPrepare4{op_type});
}} // namespace optiling
"""


def render_kernel_dispatch_skeleton(op_type: str, cases: list[TemplateCase]) -> str:
    lines = [
        f"// Generated kernel dispatch framework for {op_type}.",
        f'#include "{camel_to_snake(op_type)}.h"',
        "",
        "#define INVOKE_OP_IMPL(templateClass, ...) do { \\",
        "    templateClass<__VA_ARGS__> op; \\",
        "    op.Init(/* TODO gm inputs, workspace, tilingData */); \\",
        "    op.Process(); \\",
        "} while (0)",
        "",
        f'extern "C" __global__ __aicore__ void {camel_to_snake(op_type)}(/* TODO GM_ADDR params */)',
        "{",
        "    GET_TILING_DATA(tilingData, tiling);",
    ]
    if not cases:
        lines.extend([
            "    // TODO: add dtype macro and TILING_KEY_IS dispatch branches.",
            "    return;",
        ])
    last_dtype = None
    for case in cases:
        if case.dtype_guard != last_dtype:
            if last_dtype is not None:
                lines.append("#endif")
            if case.dtype_guard != "any" and "==" in case.dtype_guard:
                lhs, rhs = case.dtype_guard.split("==", 1)
                lines.append(f"#if ({lhs} == {rhs})")
            last_dtype = case.dtype_guard
        lines.append(f"    if (TILING_KEY_IS({case.tiling_key})) {{")
        lines.append(f"        {case.invoke}({case.args});")
        lines.append("        return;")
        lines.append("    }")
    if last_dtype is not None and last_dtype != "any":
        lines.append("#endif")
    lines.extend(["}", ""])
    return "\n".join(lines)


def write_outputs(out_dir: Path, func: ApiFunction, def_info: OpDefInfo | None, cases: list[TemplateCase],
                  aclnn_h: Path, def_cpp: Path | None, kernel_cpp: Path | None) -> None:
    op_type = derive_op_type(func, def_info)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "op_type": op_type,
        "function": {"name": func.name, "parameters": [param.__dict__ for param in func.params]},
        "op_def": None if def_info is None else {
            "inputs": [item.__dict__ for item in def_info.inputs],
            "outputs": [item.__dict__ for item in def_info.outputs],
            "attrs": [item.__dict__ for item in def_info.attrs],
            "configs": def_info.configs,
        },
        "template_cases": [case.__dict__ for case in cases],
    }
    stem = op_type
    (out_dir / f"{stem}_framework.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    (out_dir / f"{stem}_framework.md").write_text(
        render_framework_md(func, def_info, cases, aclnn_h, def_cpp, kernel_cpp), encoding="utf-8"
    )
    (out_dir / f"{stem}_aclnn_skeleton.cpp").write_text(render_aclnn_skeleton(func, def_info), encoding="utf-8")
    (out_dir / f"{stem}_tiling_skeleton.cpp").write_text(render_tiling_skeleton(op_type, def_info, cases),
                                                         encoding="utf-8")
    (out_dir / f"{stem}_kernel_dispatch_skeleton.cpp").write_text(render_kernel_dispatch_skeleton(op_type, cases),
                                                                  encoding="utf-8")
    print(f"Wrote aclnn + tiling framework to {out_dir}")


def choose_get_workspace_func(funcs: list[ApiFunction], requested: str | None) -> ApiFunction:
    if requested:
        for func in funcs:
            if func.name == requested:
                return func
        raise SystemExit(f"Function {requested!r} was not found in the header.")
    for func in funcs:
        if func.name.endswith("GetWorkspaceSize"):
            return func
    if funcs:
        return funcs[0]
    raise SystemExit("No aclnnStatus function prototype was found.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aclnn-h", required=True, type=Path, help="Path to aclnn .h interface definition.")
    parser.add_argument("--def-cpp", type=Path, help="Path to op_host/*_def.cpp.")
    parser.add_argument("--kernel-cpp", type=Path, help="Path to op_kernel/*.cpp containing template dispatch.")
    parser.add_argument("--function", help="aclnn GetWorkspaceSize function name.")
    parser.add_argument("--template-category", action="append", default=[],
                        help="Manual case: key=0;dtype=ORIG_DTYPE_X==DT_FLOAT;invoke=INVOKE_OP_IMPL;args=Op,float")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/generated_aclnn_tiling_frameworks"))
    args = parser.parse_args()

    func = choose_get_workspace_func(parse_header(args.aclnn_h), args.function)
    def_info = parse_def_cpp(args.def_cpp)
    cases = parse_kernel_templates(args.kernel_cpp)
    cases.extend(parse_manual_template_categories(args.template_category))
    write_outputs(args.out_dir, func, def_info, cases, args.aclnn_h, args.def_cpp, args.kernel_cpp)


if __name__ == "__main__":
    main()
