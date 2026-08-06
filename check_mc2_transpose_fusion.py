#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动判断 MC2 Transpose Fusion 是否成功（多算子可配）。

已内置：
  - allto_allv_quant_gmm  (AlltoAllvQuantGroupedMatMul, gmm_weight/trans_gmm_weight, flip)
  - allto_all_matmul      (AlltoAllMatmul, x2/transpose_x2, set_true)
  - matmul_allto_all      (MatmulAlltoAll, x2/transpose_x2, flip)
  - all_gather_matmul_v2  (AllGatherMatmulV2, x2/is_trans_b, flip)
  - matmul_all_reduce     (MatmulAllReduce, x2/is_trans_b, flip；x2 无 Bitcast)
  - matmul_reduce_scatter_v2 (MatmulReduceScatterV2, x2/is_trans_b, flip；可选 x2_scale)
  - quant_gmm_allto_allv  (QuantGroupedMatMulAlltoAllv, gmm_weight/trans_gmm_weight, set_true)

用法：
  python3 check_mc2_transpose_fusion.py --op allto_allv_quant_gmm --dump-dir ./pid_xxx --log run.log
  python3 check_mc2_transpose_fusion.py --list-ops
  python3 check_mc2_transpose_fusion.py --op quant_gmm_allto_allv --dump-dir ./dump --log run.log
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


TRANSPOSE_TYPES = {"Transpose", "TransposeD"}
BITCAST_TYPE = "Bitcast"

RE_PROTO_SEQ = re.compile(r"ge_proto_(\d+)_")
RE_OP_BLOCK = re.compile(r"(?:^|\n)\s*op\s*\{", re.M)
RE_NAME = re.compile(r'^\s+name:\s+"([^"]+)"', re.M)
RE_TYPE = re.compile(r'^\s+type:\s+"([^"]+)"', re.M)
RE_INPUT = re.compile(r'^\s+input:\s+"([^"]*)"', re.M)


@dataclass
class OpSpec:
    """单个算子的检测配置。"""

    key: str
    op_types: Tuple[str, ...]  # dump 中 type/name 可能的写法
    weight_input_idx: int  # 被吸收 Transpose 的输入下标（IR 顺序，主路径必检）
    weight_name: str
    attr_name: str
    # flip: 前后取反；set_true/set_false: 融合后必须为该值
    attr_mode: str
    pass_name: str  # log 里的 pass 名片段
    expect_keep_bitcast: bool = (
        True  # 若融合前是 Bitcast←Transpose，融合后是否仍直接挂 Bitcast
    )
    # 可选融合口（有 Transpose 才校验去掉；无则跳过），如 AGMMV2/MRSV2 的 x2_scale
    extra_fuse_inputs: Tuple[Tuple[int, str], ...] = ()


OP_SPECS: Dict[str, OpSpec] = {
    "allto_allv_quant_gmm": OpSpec(
        key="allto_allv_quant_gmm",
        op_types=("AlltoAllvQuantGroupedMatMul", "AlltoAllvQuantGroupedMatmul"),
        weight_input_idx=1,
        weight_name="gmm_weight",
        attr_name="trans_gmm_weight",
        attr_mode="flip",
        pass_name="AlltoAllvQuantGroupedMatMulTransposeFusionPass",
        expect_keep_bitcast=True,
    ),
    "allto_all_matmul": OpSpec(
        key="allto_all_matmul",
        op_types=("AlltoAllMatmul",),
        weight_input_idx=1,
        weight_name="x2",
        attr_name="transpose_x2",
        attr_mode="set_true",  # ES/canndev：吸收后强制 true
        pass_name="AllToAllMatmulTransposeA5FusionPass",
        expect_keep_bitcast=True,
    ),
    "matmul_allto_all": OpSpec(
        key="matmul_allto_all",
        op_types=("MatmulAlltoAll",),
        weight_input_idx=1,
        weight_name="x2",
        attr_name="transpose_x2",
        attr_mode="flip",
        pass_name="MatmulAllToAllTransposeA5FusionPass",
        expect_keep_bitcast=True,  # canndev/ES：x2 支持 Bitcast←Transpose
    ),
    "all_gather_matmul_v2": OpSpec(
        key="all_gather_matmul_v2",
        op_types=("AllGatherMatmulV2",),
        weight_input_idx=1,
        weight_name="x2",
        attr_name="is_trans_b",
        attr_mode="flip",
        pass_name="AllGatherMatmulV2TransposeA5FusionPass",
        expect_keep_bitcast=True,
        extra_fuse_inputs=((4, "x2_scale"),),  # MC2V2InputIdx::K_X2SCALE
    ),
    "matmul_all_reduce": OpSpec(
        key="matmul_all_reduce",
        op_types=("MatmulAllReduce",),
        weight_input_idx=1,  # MC2InputIdx::K_X2
        weight_name="x2",
        attr_name="is_trans_b",
        attr_mode="flip",
        pass_name="MatmulAllReduceTransposeA5FusionPass",
        # canndev/ES：x2 仅支持 Transpose/D，不支持 Bitcast←Transpose；Bitcast 只在 dequant
        expect_keep_bitcast=False,
    ),
    "matmul_reduce_scatter_v2": OpSpec(
        key="matmul_reduce_scatter_v2",
        op_types=("MatmulReduceScatterV2",),
        weight_input_idx=1,
        weight_name="x2",
        attr_name="is_trans_b",
        attr_mode="flip",
        pass_name="MatmulReduceScatterV2TransposeA5FusionPass",
        expect_keep_bitcast=True,  # canndev/ES：x2 / x2_scale 均支持 Bitcast←Transpose
        extra_fuse_inputs=((4, "x2_scale"),),  # MC2V2InputIdx::K_X2SCALE
    ),
    "quant_gmm_allto_allv": OpSpec(
        key="quant_gmm_allto_allv",
        op_types=("QuantGroupedMatMulAlltoAllv",),
        weight_input_idx=1,  # gmm_weight
        weight_name="gmm_weight",
        attr_name="trans_gmm_weight",
        attr_mode="set_true",  # canndev：吸收后强制 true（镜像算子 allto_allv_quant_gmm 是 flip）
        pass_name="QuantGroupedMatMulAlltoAllvTransposeFusionPass",
        expect_keep_bitcast=True,
        extra_fuse_inputs=(
            (3, "gmm_weight_scale"),
            (7, "mm_weight"),
            (9, "mm_weight_scale"),
        ),
    ),
}


@dataclass
class OpInfo:
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)


@dataclass
class GraphSnapshot:
    path: str
    seq: int
    stage: str
    mc2: OpInfo
    ops: Dict[str, OpInfo]
    attr_value: Optional[bool]


def _attr_regex(attr_name: str) -> re.Pattern:
    return re.compile(
        rf'key:\s*"{re.escape(attr_name)}"\s*(?:\n\s*)+value\s*\{{\s*(?:\n\s*)*(?:b:\s*(true|false)|i:\s*(\d+))',
        re.M,
    )


def _parse_bool_attr(text: str, attr_name: str) -> Optional[bool]:
    m = _attr_regex(attr_name).search(text)
    if not m:
        return None
    if m.group(1) is not None:
        return m.group(1) == "true"
    if m.group(2) is not None:
        return m.group(2) != "0"
    return None


def _split_op_blocks(text: str) -> List[str]:
    parts = RE_OP_BLOCK.split(text)
    return parts[1:] if len(parts) > 1 else []


def _parse_op_block(block: str) -> Optional[OpInfo]:
    head = block[:4000]
    name_m = RE_NAME.search(head)
    type_m = RE_TYPE.search(head)
    if not name_m or not type_m:
        return None
    return OpInfo(
        name=name_m.group(1), op_type=type_m.group(1), inputs=RE_INPUT.findall(head)
    )


def _peer_name(tensor_ref: str) -> Optional[str]:
    if not tensor_ref:
        return None
    return tensor_ref.split(":", 1)[0]


def _text_has_op(text: str, spec: OpSpec) -> bool:
    return any(t in text for t in spec.op_types)


def _is_target_op(info: OpInfo, spec: OpSpec) -> bool:
    return info.op_type in spec.op_types or info.name in spec.op_types


def resolve_weight_chain(
    ops: Dict[str, OpInfo], mc2: OpInfo, weight_idx: int, max_depth: int = 4
) -> List[str]:
    if len(mc2.inputs) <= weight_idx:
        return []
    chain: List[str] = []
    cur = _peer_name(mc2.inputs[weight_idx])
    seen = set()
    depth = 0
    while cur and cur not in seen and depth < max_depth:
        seen.add(cur)
        op = ops.get(cur)
        if op is None:
            chain.append(f"UNKNOWN({cur})")
            break
        chain.append(op.op_type)
        if not op.inputs:
            break
        if op.op_type in ("Data", "PartitionedCall") or op.op_type.startswith("arg"):
            break
        nxt = _peer_name(op.inputs[0])
        if nxt is None:
            break
        cur = nxt
        depth += 1
    return chain


def has_transpose_on_weight(chain: List[str]) -> bool:
    return any(t in TRANSPOSE_TYPES for t in chain)


def has_bitcast_direct(chain: List[str]) -> bool:
    return bool(chain) and chain[0] == BITCAST_TYPE


def parse_dump_file(path: str, spec: OpSpec) -> Optional[GraphSnapshot]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    if not _text_has_op(text, spec):
        return None

    seq_m = RE_PROTO_SEQ.search(os.path.basename(path))
    seq = int(seq_m.group(1)) if seq_m else -1
    base = os.path.basename(path)
    stage = base
    if "_graph_" in base:
        stage = base.split("_graph_", 1)[-1]
        stage = re.sub(r"^\d+_", "", stage)
        stage = re.sub(r"\.(txt|pbtxt)$", "", stage)

    ops: Dict[str, OpInfo] = {}
    mc2: Optional[OpInfo] = None
    mc2_block: Optional[str] = None
    for block in _split_op_blocks(text):
        info = _parse_op_block(block)
        if info is None:
            continue
        ops[info.name] = info
        if _is_target_op(info, spec):
            mc2 = info
            mc2_block = block

    if mc2 is None:
        return None

    attr_val = _parse_bool_attr(mc2_block[:8000], spec.attr_name) if mc2_block else None
    if attr_val is None:
        attr_val = _parse_bool_attr(text, spec.attr_name)

    return GraphSnapshot(
        path=path, seq=seq, stage=stage, mc2=mc2, ops=ops, attr_value=attr_val
    )


def collect_snapshots(
    dump_dir: str, spec: OpSpec
) -> Tuple[List[GraphSnapshot], List[str]]:
    snaps: List[GraphSnapshot] = []
    all_files = 0
    txt_like = 0
    candidates: List[str] = []
    for root, _, files in os.walk(dump_dir):
        for name in files:
            all_files += 1
            lower = name.lower()
            if not (lower.endswith(".txt") or lower.endswith(".pbtxt")):
                continue
            txt_like += 1
            candidates.append(os.path.join(root, name))

    contain = 0
    parse_fail = 0
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(2 * 1024 * 1024)
            if not _text_has_op(head, spec):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if not _text_has_op(text, spec):
                    continue
            contain += 1
            snap = parse_dump_file(path, spec)
            if snap is None:
                parse_fail += 1
            else:
                snaps.append(snap)
        except OSError:
            continue

    diag = [
        f"op={spec.key} types={list(spec.op_types)} weight_idx={spec.weight_input_idx}({spec.weight_name})",
        f"attr={spec.attr_name} mode={spec.attr_mode}",
        f"walk 文件总数={all_files}",
        f"txt/pbtxt 数={txt_like}",
        f"待解析候选={len(candidates)}",
        f"含算子关键字文件数={contain}",
        f"关键字命中但解析失败={parse_fail}",
        f"成功解析快照={len(snaps)}",
    ]
    if contain == 0 and candidates:
        diag.append("样例文件: " + ", ".join(candidates[:3]))
        diag.append(f"提示: grep -l '{spec.op_types[0]}' <dump-dir>/*.txt | head")
    snaps.sort(key=lambda s: (s.seq, s.path))
    return snaps, diag


def check_log(log_path: str, spec: OpSpec) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError as e:
        return False, [f"无法读取 log: {e}"]

    related = [ln for ln in text.splitlines() if spec.pass_name in ln]
    if not related:
        snake = spec.key.replace("-", "_")
        related = [ln for ln in text.splitlines() if snake in ln and "FusionPass" in ln]
    if not related:
        msgs.append(f"log 中未找到 pass 相关行: {spec.pass_name}")
        return False, msgs

    has_meet = any(
        "Enter MeetRequirements" in ln or "MeetRequirements" in ln for ln in related
    )
    has_found = any("Found One pattern that meets requirements" in ln for ln in related)
    has_repl = any("Enter Replacement" in ln for ln in related)
    has_fail = any(
        ("Replacement failed" in ln)
        or ("InferShapeReplaceGraph failed" in ln)
        or ("BuildReplaceGraph failed" in ln)
        for ln in related
    )
    msgs.append(
        f"MeetRequirements={has_meet}, FoundPattern={has_found}, Replacement={has_repl}, Failed={has_fail}"
    )
    # FoundPattern 在部分 pass 日志文案可能不同，放宽：有 Meet + Replacement 且无 fail 也算
    ok = has_repl and not has_fail and (has_found or has_meet)
    return ok, msgs


def _check_attr(
    before: Optional[bool], after: Optional[bool], mode: str, attr_name: str
) -> Tuple[bool, str]:
    if after is None:
        return False, f"无法解析融合后 {attr_name}"
    if mode == "flip":
        # 早期 dump（如 GraphPreRunBegin）常未写出 bool attr；GE 默认 false，与 flip→True 一致
        if before is None:
            if after is True:
                return (
                    True,
                    f"{attr_name} 融合前未写出(按默认 False)，融合后 True（与 flip 一致）",
                )
            return (
                False,
                f"无法解析融合前 {attr_name}，且融合后为 {after}（无法确认 flip）",
            )
        if before == after:
            return False, f"{attr_name} 未取反 (均为 {before})"
        return True, f"{attr_name} 已取反: {before} -> {after}"
    if mode == "set_true":
        if after is not True:
            return False, f"{attr_name} 融合后应为 True，实际 {after}"
        return True, f"{attr_name} 融合后为 True (before={before})"
    if mode == "set_false":
        if after is not False:
            return False, f"{attr_name} 融合后应为 False，实际 {after}"
        return True, f"{attr_name} 融合后为 False (before={before})"
    return False, f"未知 attr_mode={mode}"


def _analyze_port(
    ops: Dict[str, OpInfo], mc2: OpInfo, input_idx: int
) -> Tuple[List[str], bool, bool]:
    chain = resolve_weight_chain(ops, mc2, input_idx)
    return chain, has_transpose_on_weight(chain), has_bitcast_direct(chain)


def _check_extra_fuse_ports(
    before_snap: GraphSnapshot,
    after_snap: GraphSnapshot,
    spec: OpSpec,
) -> Tuple[bool, List[str]]:
    """可选口：融合前有 Transpose 则融合后必须去掉；Bitcast←Transpose 时按 expect_keep_bitcast。"""
    msgs: List[str] = []
    ok = True
    for idx, name in spec.extra_fuse_inputs:
        b_chain, b_trans, b_bitcast = _analyze_port(
            before_snap.ops, before_snap.mc2, idx
        )
        a_chain, a_trans, a_bitcast = _analyze_port(after_snap.ops, after_snap.mc2, idx)
        if not b_trans:
            msgs.append(
                f"{name}[{idx}] 融合前无 Transpose（跳过可选口校验）链={b_chain}"
            )
            continue
        if a_trans:
            ok = False
            msgs.append(f"{name}[{idx}] 融合后仍有 Transpose 链={a_chain}")
        else:
            msgs.append(f"{name}[{idx}] 融合后已无 Transpose 链={a_chain}")
        if spec.expect_keep_bitcast and b_bitcast:
            if not a_bitcast:
                ok = False
                msgs.append(
                    f"{name}[{idx}] 融合前 Bitcast←Transpose，融合后未保留 Bitcast"
                )
            else:
                msgs.append(f"{name}[{idx}] 融合后仍保留 Bitcast 直接前驱")
    return ok, msgs


def judge_fusion(snaps: List[GraphSnapshot], spec: OpSpec) -> Tuple[bool, str, dict]:
    if not snaps:
        return False, f"dump 中未找到含 {spec.op_types[0]} 的图", {}

    analyzed = []
    for s in snaps:
        chain = resolve_weight_chain(s.ops, s.mc2, spec.weight_input_idx)
        entry = {
            "seq": s.seq,
            "stage": s.stage,
            "path": s.path,
            "weight_inputs": s.mc2.inputs[: max(5, spec.weight_input_idx + 1)],
            "weight_chain": chain,
            "has_transpose": has_transpose_on_weight(chain),
            "bitcast_direct": has_bitcast_direct(chain),
            "attr_value": s.attr_value,
            "snap": s,
            "extra_ports": {},
        }
        for idx, name in spec.extra_fuse_inputs:
            e_chain, e_trans, e_bitcast = _analyze_port(s.ops, s.mc2, idx)
            entry["extra_ports"][name] = {
                "idx": idx,
                "chain": e_chain,
                "has_transpose": e_trans,
                "bitcast_direct": e_bitcast,
            }
        analyzed.append(entry)

    # 优先选「仍有 Transpose 且已能解析 attr」的 BEFORE，避免 GraphPreRunBegin 缺 attr 误判
    before = next(
        (a for a in analyzed if a["has_transpose"] and a["attr_value"] is not None),
        None,
    )
    if before is None:
        before = next((a for a in analyzed if a["has_transpose"]), None)
    if before is None:
        return (
            False,
            f"全程未在 {spec.weight_name}(input[{spec.weight_input_idx}]) 前驱链上看到 Transpose/TransposeD",
            {"analyzed": analyzed},
        )

    after_candidates = [
        a for a in analyzed if a["seq"] > before["seq"] and not a["has_transpose"]
    ]
    if not after_candidates:
        return (
            False,
            f"融合前({before['stage']}) 链={before['weight_chain']}，后续 dump 未见去掉 Transpose",
            {"before": before, "analyzed": analyzed},
        )
    # 优先选能解析 attr 的 AFTER（末帧可能被 lowering 改写）
    after_with_attr = [a for a in after_candidates if a["attr_value"] is not None]
    after = after_with_attr[-1] if after_with_attr else after_candidates[-1]

    reasons: List[str] = []
    ok = True

    if after["has_transpose"]:
        ok = False
        reasons.append("融合后仍存在 Transpose")
    else:
        reasons.append(f"融合后 {spec.weight_name} 链已无 Transpose")

    if (
        spec.expect_keep_bitcast
        and before["bitcast_direct"]
        and before["has_transpose"]
    ):
        if not after["bitcast_direct"]:
            ok = False
            reasons.append("融合前为 Bitcast←Transpose，融合后未保留直接 Bitcast 前驱")
        else:
            reasons.append("融合后仍保留 Bitcast 直接前驱")

    attr_ok, attr_msg = _check_attr(
        before["attr_value"], after["attr_value"], spec.attr_mode, spec.attr_name
    )
    reasons.append(attr_msg)
    if not attr_ok:
        ok = False

    if spec.extra_fuse_inputs:
        extra_ok, extra_msgs = _check_extra_fuse_ports(
            before["snap"], after["snap"], spec
        )
        reasons.extend(extra_msgs)
        if not extra_ok:
            ok = False

    return (
        ok,
        "; ".join(reasons),
        {"before": before, "after": after, "reasons": reasons},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check MC2 transpose fusion success (multi-op)"
    )
    parser.add_argument(
        "--op",
        default="allto_allv_quant_gmm",
        choices=sorted(OP_SPECS.keys()),
        help="算子配置",
    )
    parser.add_argument("--dump-dir", default=None, help="GE dump 目录")
    parser.add_argument("--log", default=None, help="可选 run.log")
    parser.add_argument("--strict-log", action="store_true", help="log 失败则整体 FAIL")
    parser.add_argument(
        "--list-ops", action="store_true", help="列出支持的算子配置后退出"
    )
    args = parser.parse_args()

    if args.list_ops:
        for k, s in OP_SPECS.items():
            extras = ",".join(f"{n}[{i}]" for i, n in s.extra_fuse_inputs) or "-"
            print(
                f"{k}: type={s.op_types[0]}, weight={s.weight_name}[{s.weight_input_idx}], "
                f"attr={s.attr_name}({s.attr_mode}), extra={extras}, pass={s.pass_name}"
            )
        return 0

    if not args.dump_dir:
        print("[FAIL] 请提供 --dump-dir")
        return 2

    spec = OP_SPECS[args.op]
    dump_dir = os.path.abspath(args.dump_dir)
    if not os.path.isdir(dump_dir):
        print(f"[FAIL] dump-dir 不存在: {dump_dir}")
        return 2

    snaps, diag = collect_snapshots(dump_dir, spec)
    dump_ok, dump_msg, detail = judge_fusion(snaps, spec)

    print("==== Dump 扫描诊断 ====")
    print(f"dump-dir: {dump_dir}")
    for d in diag:
        print(d)

    print("==== Dump 判定 ====")
    print(f"快照数: {len(snaps)}")
    if "before" in detail:
        b = detail["before"]
        print(f"BEFORE seq={b['seq']} stage={b['stage']}")
        print(f"  path={b['path']}")
        print(f"  weight_inputs={b['weight_inputs']}")
        print(f"  weight_chain={b['weight_chain']}")
        print(f"  {spec.attr_name}={b['attr_value']}")
        for name, port in b.get("extra_ports", {}).items():
            print(
                f"  {name}[{port['idx']}] chain={port['chain']} transpose={port['has_transpose']}"
            )
    if "after" in detail:
        a = detail["after"]
        print(f"AFTER  seq={a['seq']} stage={a['stage']}")
        print(f"  path={a['path']}")
        print(f"  weight_inputs={a['weight_inputs']}")
        print(f"  weight_chain={a['weight_chain']}")
        print(f"  {spec.attr_name}={a['attr_value']}")
        for name, port in a.get("extra_ports", {}).items():
            print(
                f"  {name}[{port['idx']}] chain={port['chain']} transpose={port['has_transpose']}"
            )
    print(f"结果: {'PASS' if dump_ok else 'FAIL'} | {dump_msg}")

    log_ok = True
    if args.log:
        print("==== Log 判定 ====")
        log_ok, log_msgs = check_log(args.log, spec)
        for m in log_msgs:
            print(m)
        print(f"结果: {'PASS' if log_ok else 'FAIL'}")

    if dump_ok and (log_ok or not args.strict_log):
        if args.log and not log_ok and not args.strict_log:
            print("==== 总结: PASS（dump 通过；log 未通过但未 --strict-log）====")
            return 0
        print("==== 总结: PASS ====")
        return 0

    print("==== 总结: FAIL ====")
    return 1


if __name__ == "__main__":
    sys.exit(main())
