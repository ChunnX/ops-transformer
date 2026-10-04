"""Single-op profiling recipe for the SageAttention2 PyTorch path on Atlas 950PR.

取 ./benchmarks/cases.py 的一个 case（默认 P-01），经 sageattn_qk_int8_pv_fp8_asc
（fp32+fp16 交付默认档）在真机上跑 warmup + 受测循环，用 torch_npu.profiler（kineto）
采集算子级耗时，导出 chrome trace 并打印 op 级汇总表。本脚本是 PyTorch 路径的
单算子 profile 配方，与 C++ aclnn 基准工具 ./benchmarks/benchmark_aclnn.cpp 互补，
不导入、不依赖该二进制。

真机用法（算子目录执行）：

    python3 benchmarks/profile_single_op.py --case P-01 --warmup 10 --steps 20 \
        --trace build/profile_P-01.json

msprof CLI 整进程附加（可选；受测循环保持小步数，采集细节以验收环境 msprof 文档为准）：

    msprof --application="python3 benchmarks/profile_single_op.py --case P-01 \
        --warmup 5 --steps 10 --no-trace" --output=build/msprof_p01

无 torch/torch_npu 时打印跳过信息并以退出码 0 结束（CI-off 机器冒烟）；
在 NPU 上为严格模式，任何失败非零退出。

------------------------------------------------------------------------------------
M6 微架构自动化探针面（task/todo.md 阶段 M6 探针 1-8）：

    python3 benchmarks/profile_single_op.py --probe {1..8|all} [--case P-xx]

不带 --probe 时行为与上方默认路径完全一致。探针 1-7 为真机工具链（msprof PMU /
msprof-analyze / npusim / mssanitizer），本环境缺工具或 NPU 时打印明确提示并以专用
退出码 3 跳过（非失败崩溃）；环境齐备时原样执行 todo 原文命令并回传工具退出码。
探针 8（ccec 宏探测 + llvm-objdump 反汇编核验）在 BiSheng 容器内可本地全量执行：

    bash docker/sage run python3 benchmarks/profile_single_op.py --probe 8

探针产物默认落 build/msprof/<case>/、build/advisor/<case>/、build/sim/<case>/
（可用 --probe-build-root 改根）。探针 1 的 msprof 输出目录固定为
build/msprof/<case>/sage，与探针 5 的 advisor 输入 -d 路径（todo 原文）闭环衔接。
探针 8 的结构化结论（命中/未命中 + 反汇编证据行号与样例行）落
build/msprof/<case>/probe8/probe8_report.json，绝不编造命中：反汇编无法定位的指令
记号如实输出「未命中」，并附检索用候选助记符/内置函数名清单。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

BENCH_DIR = Path(__file__).resolve().parent
OP_DIR = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from cases import BenchmarkCase, get_case  # noqa: E402

# 探针环境不可用（缺真机工具/NPU 设备）时的专用退出码：明确跳过而非失败崩溃
PROBE_EXIT_UNAVAILABLE = 3

# 探针 1-4 的 msprof op --aic-metrics 取值（todo 原文，逐字对齐）与输出子目录
AIC_METRICS = {
    1: ("PipeUtilization", "sage"),           # Cube/Vector/FixPipe/MTE2/MTE3 周期占比 → 探针 5 的 -d 输入
    2: ("L2Cache", "sage_l2cache"),           # l2_hit_ratio、读写带宽、128B Sector 命中
    3: ("ResourceConflictRatio", "sage_conflict"),  # L1 双 Bank 冲突、UB 争抢
    4: ("Roofline", "sage_roofline"),         # 计算强度与 950PR Roofline 标点
}

PROBE_TITLES = {
    1: "执行流水与气泡探测（msprof op --aic-metrics=PipeUtilization）",
    2: "L2 命中与带宽探测（msprof op --aic-metrics=L2Cache）",
    3: "片上存储冲突探测（msprof op --aic-metrics=ResourceConflictRatio）",
    4: "算力带宽天花板判定（msprof op --aic-metrics=Roofline）",
    5: "自动化专家诊断建议（msprof-analyze advisor）",
    6: "SoC 周期级指令流水全真透视（npusim record）",
    7: "微架构非对齐与硬件竞争检测（mssanitizer memcheck+racecheck）",
    8: "编译器指令集与反汇编自动化探测（ccec -dM + llvm-objdump -d）",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one benchmark case of sageattn_qk_int8_pv_fp8_asc with torch_npu.profiler; "
                    "or run the M6 microarch probes 1-8 via --probe."
    )
    parser.add_argument("--case", type=str, default="P-01", help="Case name from benchmarks/cases.py (default P-01)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations outside the profiler (default 10)")
    parser.add_argument("--steps", type=int, default=20, help="Profiled iterations (default 20)")
    parser.add_argument("--trace", type=str, default="", help="Chrome trace output path (default build/profile_<case>.json)")
    parser.add_argument("--no-trace", action="store_true", help="Skip chrome trace export (for msprof attach runs)")
    # ---- M6 探针面（与默认行为共存：不带 --probe 时完全走原单用例采集路径）----
    parser.add_argument(
        "--probe", type=str, default="", metavar="{1..8|all}",
        help="Run M6 microarch probe(s) from task/todo.md instead of the default single-op profiling "
             "(1=PipeUtilization 2=L2Cache 3=ResourceConflictRatio 4=Roofline 5=advisor 6=npusim "
             "7=mssanitizer 8=ccec+llvm-objdump, or all)",
    )
    parser.add_argument(
        "--probe-build-root", type=str, default="",
        help="Root dir for probe outputs (default <op>/build; probes land in msprof/ advisor/ sim/ under it)",
    )
    parser.add_argument(
        "--kernel-name", type=str, default="SageAttention2",
        help="Kernel name filter for msprof op probes 1-4 (todo 原文: SageAttention2)",
    )
    parser.add_argument(
        "--npusim-app", type=str, default="./test_aclnn_sageattn",
        help="Application binary for npusim probe 6 (todo 原文: ./test_aclnn_sageattn)",
    )
    parser.add_argument(
        "--probe8-object", type=str, default="build/sageattention2_bisheng.o",
        help="Device object for probe 8 disassembly verification (default run.sh kernel 产物)",
    )
    return parser


def build_inputs(case: BenchmarkCase, torch):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]
    q = torch.randn(case.batch, case.heads, case.seq, case.dim, dtype=dtype, device="npu")
    k = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    v = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    return q, k, v


# ==================================================================================
# M6 探针公共设施
# ==================================================================================

def probe_build_root(args: argparse.Namespace) -> Path:
    """探针产物根目录：默认 <算子目录>/build，可经 --probe-build-root 覆盖。"""
    root = Path(args.probe_build_root) if args.probe_build_root else OP_DIR / "build"
    if not root.is_absolute():
        root = OP_DIR / root
    return root


def find_tool(name: str) -> Optional[str]:
    """按 PATH → ${ASCEND_HOME_PATH}/bin 顺序解析工具真实可执行路径，返回绝对路径或 None。"""
    found = shutil.which(name)
    if found:
        return found
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        cand = Path(ascend_home) / "bin" / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def npu_device_ready() -> bool:
    """检测 NPU 设备存在性：昇腾 davinci 设备节点或 npu-smi 任一可用即认为就绪。"""
    for dev in ("/dev/davinci_manager", "/dev/davinci0"):
        if Path(dev).exists():
            return True
    return find_tool("npu-smi") is not None


def run_streaming(cmd: List[str], cwd: Path, log_path: Path) -> int:
    """执行真机工具命令：stdout/stderr 实时回显并落日志文件，回传工具退出码。"""
    print(f"[probe-cmd] {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# {shlex.join(cmd)}\n# cwd: {cwd}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.stdout.close()
        return proc.wait()


def capture_cmd(cmd: List[str], stdin_devnull: bool = False) -> Tuple[int, str, str]:
    """静默捕获式执行（探针 8 宏探测用），回传 (rc, stdout, stderr)。"""
    proc = subprocess.run(
        cmd, cwd=str(OP_DIR), stdin=subprocess.DEVNULL if stdin_devnull else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def skip_unavailable(probe_no: int, reasons: List[str]) -> int:
    """打印明确的「本环境不可用，需真机」信息并返回专用退出码 3。"""
    print(f"[probe-{probe_no}] {PROBE_TITLES.get(probe_no, '')}")
    for reason in reasons:
        print(f"[probe-{probe_no}] [SKIP] {reason}")
    print(f"[probe-{probe_no}] 本环境不可用，需真机（Atlas 950PR + CANN 工具链）；以退出码 "
          f"{PROBE_EXIT_UNAVAILABLE} 跳过。")
    return PROBE_EXIT_UNAVAILABLE


# ==================================================================================
# 探针 1-4：msprof op PMU 采集（真机）
# ==================================================================================

def run_msprof_metric_probe(probe_no: int, args: argparse.Namespace) -> int:
    metric, subdir = AIC_METRICS[probe_no]
    msprof = find_tool("msprof")
    reasons = []
    if msprof is None:
        reasons.append("msprof 可执行文件未找到（PATH 与 ${ASCEND_HOME_PATH}/bin 均无），真机 CANN 工具链必备。")
    if not npu_device_ready():
        reasons.append("未检测到 NPU 设备（无 /dev/davinci* 节点且无 npu-smi），PMU 采集必须真机执行。")
    if reasons:
        return skip_unavailable(probe_no, reasons)

    case = get_case(args.case)  # 校验 case 名合法（与默认路径同一用例表）
    outdir = probe_build_root(args) / "msprof" / args.case / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    # todo 原文语义：msprof op --aic-metrics=<M> --kernel-name=SageAttention2 ...
    # 「...」补全为受测应用（本脚本自身小步数采集，不导出 trace）与 --output 目录。
    app = (f"python3 benchmarks/profile_single_op.py --case {case.name} "
           f"--warmup 5 --steps 10 --no-trace")
    cmd = [
        msprof, "op",
        f"--aic-metrics={metric}",
        f"--kernel-name={args.kernel_name}",
        f"--application={app}",
        f"--output={outdir}",
    ]
    log = probe_build_root(args) / "msprof" / args.case / f"probe{probe_no}_{metric}.log"
    print(f"[probe-{probe_no}] {PROBE_TITLES[probe_no]} | case={case.name}")
    rc = run_streaming(cmd, OP_DIR, log)
    print(f"[probe-{probe_no}] msprof 退出码 {rc}；日志 {log}；产物目录 {outdir}")
    return rc


# ==================================================================================
# 探针 5：msprof-analyze advisor 专家诊断（真机）
# ==================================================================================

def run_advisor_probe(args: argparse.Namespace) -> int:
    tool = find_tool("msprof-analyze")
    reasons = []
    if tool is None:
        reasons.append("msprof-analyze 未安装/未找到（PATH 与 ${ASCEND_HOME_PATH}/bin 均无）；"
                       "按 todo 需先在真机安装 msprof-analyze（MindStudio Insight 组件）。")
    indir = probe_build_root(args) / "msprof" / args.case / "sage"
    if not indir.is_dir():
        reasons.append(f"expert 数据目录 {indir} 不存在；请先执行 --probe 1 生成 msprof PipeUtilization 数据。")
    if reasons:
        return skip_unavailable(5, reasons)

    outdir = probe_build_root(args) / "advisor" / args.case
    outdir.mkdir(parents=True, exist_ok=True)
    # todo 原文：msprof-analyze advisor -d build/msprof/P-02/sage --out-dir build/advisor/P-02
    cmd = [tool, "advisor", "-d", str(indir), "--out-dir", str(outdir)]
    log = probe_build_root(args) / "msprof" / args.case / "probe5_advisor.log"
    print(f"[probe-5] {PROBE_TITLES[5]} | case={args.case}")
    rc = run_streaming(cmd, OP_DIR, log)
    print(f"[probe-5] msprof-analyze 退出码 {rc}；建议报告目录 {outdir}")
    return rc


# ==================================================================================
# 探针 6：npusim 周期级指令流水仿真（无硬件仿真环境）
# ==================================================================================

def run_npusim_probe(args: argparse.Namespace) -> int:
    tool = find_tool("npusim")
    reasons = []
    if tool is None:
        reasons.append("npusim 可执行文件未找到（PATH 与 ${ASCEND_HOME_PATH}/bin 均无）。")
    app = Path(args.npusim_app)
    if app.is_absolute():
        app_resolved = app
    else:
        app_resolved = OP_DIR / app
        if not app_resolved.is_file() and shutil.which(str(app)):
            app_resolved = Path(str(shutil.which(str(app))))
    if not app_resolved.is_file():
        reasons.append(f"受测应用 {app_resolved} 不存在；请先编译/部署 {args.npusim_app}（真机仿真用例宿主）。")
    if reasons:
        return skip_unavailable(6, reasons)

    outdir = probe_build_root(args) / "sim" / args.case
    outdir.mkdir(parents=True, exist_ok=True)
    # todo 原文：npusim record ./test_aclnn_sageattn -s Ascend950 --gen-report -o build/sim/P-02
    cmd = [tool, "record", str(app_resolved), "-s", "Ascend950", "--gen-report", "-o", str(outdir)]
    log = probe_build_root(args) / "msprof" / args.case / "probe6_npusim.log"
    print(f"[probe-6] {PROBE_TITLES[6]} | case={args.case}")
    rc = run_streaming(cmd, OP_DIR, log)
    trace_hits = sorted(outdir.rglob("trace_core0.json"))
    if trace_hits:
        print(f"[probe-6] trace_core0.json 已导出：{trace_hits[0]}（Chrome 打开可纳秒级查看指令气泡）")
    else:
        print(f"[probe-6] [WARN] 产物目录中未发现 trace_core0.json，请检查 npusim 日志 {log}")
    return rc


# ==================================================================================
# 探针 7：mssanitizer 非对齐与 RAW/WAR 竞争检测（真机）
# ==================================================================================

def run_sanitizer_probe(args: argparse.Namespace) -> int:
    tool = find_tool("mssanitizer")
    reasons = []
    if tool is None:
        reasons.append("mssanitizer 可执行文件未找到（PATH 与 ${ASCEND_HOME_PATH}/bin 均无），真机 CANN 工具链必备。")
    if not npu_device_ready():
        reasons.append("未检测到 NPU 设备（无 /dev/davinci* 节点且无 npu-smi），硬件竞争检测必须真机执行。")
    if shutil.which("python3") is None:
        reasons.append("python3 未找到，无法承载受测脚本 benchmarks/run_validation_performance.py。")
    if reasons:
        return skip_unavailable(7, reasons)

    target = OP_DIR / "benchmarks" / "run_validation_performance.py"
    if not target.is_file():
        return skip_unavailable(7, [f"受测脚本 {target} 不存在。"])
    # todo 原文：mssanitizer --tool=memcheck --tool=racecheck -- python3 benchmarks/run_validation_performance.py
    cmd = [tool, "--tool=memcheck", "--tool=racecheck", "--",
           "python3", "benchmarks/run_validation_performance.py"]
    log = probe_build_root(args) / "msprof" / args.case / "probe7_sanitizer.log"
    print(f"[probe-7] {PROBE_TITLES[7]}（未对齐 128B Sector 访问与 RAW/WAR 管线竞争检测）")
    rc = run_streaming(cmd, OP_DIR, log)
    print(f"[probe-7] mssanitizer 退出码 {rc}；日志 {log}")
    return rc


# ==================================================================================
# 探针 8：ccec 预定义宏探测 + llvm-objdump 反汇编核验（BiSheng 容器内本地全量执行）
# ==================================================================================

# 指令记号候选清单：反汇编解码器缺失时供真机/解码工具检索（如实标注，绝不编造命中）
PROBE8_CANDIDATES = {
    "nddma": ["nddma", "nddma_out_to_ub", "nddma_ub_to_ub", "set_loop0_stride_nddma",
              "set_pad_cnt_nddma", "__builtin_cce_nddma_"],
    "fixpipe": ["fixp", "fixpipe", "set_fixp_addr", "set_fixp_max_cfg", "set_fixp_nz_para",
                "__fbuf_alloc", "cce_fixpipe_buff"],
    "bufferid": ["get_buf", "rel_buf", "bufid", "buf_id", "__builtin_cce_get_buf"],
}

# 内置函数/宏层「950 原生指令开关」检索模式（对 save-temps 预处理体与 -dM 宏表扫描）
PROBE8_INTRINSIC_PATTERNS = {
    "nddma": [r"__builtin_cce_[a-z_0-9]*nddma[a-z_0-9]*", r"\bnddma_[a-z_0-9_]+\("],
    "fixpipe": [r"\bset_fixp_[a-z_0-9]+\(", r"__fbuf_alloci?\b", r"cce_fixpipe_buff"],
    "bufferid": [r"__builtin_cce_get_buf\b", r"\brel_buf\b", r"\bUseBufIdSync\b"],
}


class Probe8Report:
    """探针 8 结构化结论收集器：verdict ∈ 命中 / 未命中 / 跳过，证据带文件与 1 起始行号。"""

    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        self.items: List[Dict[str, object]] = []

    def add(self, item: str, verdict: str, evidence: Optional[List[Dict[str, object]]] = None,
            note: str = "", candidates: Optional[List[str]] = None) -> None:
        entry: Dict[str, object] = {"item": item, "verdict": verdict}
        if evidence:
            entry["evidence"] = evidence
        if note:
            entry["note"] = note
        if candidates:
            entry["candidates"] = candidates
        self.items.append(entry)
        ev_mark = f"（证据 {len(evidence)} 条）" if evidence else ""
        print(f"    - {item}: {verdict}{ev_mark}")

    def evidence(self, path: Path, line_no: int, text: str) -> Dict[str, object]:
        return {"file": str(path), "line": line_no, "text": text.strip()[:200]}

    def save(self, meta: Dict[str, object]) -> Path:
        doc = {"meta": meta, "findings": self.items}
        path = self.outdir / "probe8_report.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def grep_evidence(path: Path, patterns: List[str], max_hits: int = 3) -> List[Dict[str, object]]:
    """在文本文件中检索候选记号，返回前 max_hits 条证据（1 起始行号 + 样例行）。"""
    if not path.is_file():
        return []
    rxes = [re.compile(p) for p in patterns]
    hits: List[Dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for idx, text in enumerate(lines, start=1):
        if any(rx.search(text) for rx in rxes):
            hits.append({"file": str(path), "line": idx, "text": text.strip()[:200]})
            if len(hits) >= max_hits:
                break
    return hits


def _probe8_tool_inventory(rep: Probe8Report) -> Dict[str, Optional[str]]:
    print("[probe-8] 工具链真实可执行名与路径考证：")
    tools: Dict[str, Optional[str]] = {}
    for name in ("ccec", "bisheng", "llvm-objdump", "llvm-objcopy"):
        tools[name] = find_tool(name)
        rep.add(f"tool:{name}", "命中" if tools[name] else "未命中",
                [rep.evidence(Path(tools[name]), 0, tools[name])] if tools[name] else [],
                note="PATH 与 ${ASCEND_HOME_PATH}/bin 解析结果（如实工具名）")
    return tools


def _probe8_macro_probes(rep: Probe8Report, tools: Dict[str, Optional[str]]) -> None:
    """todo 原文命令 → ccec 原生拼写 → bisheng dav-3510 设备模式，逐级如实落账。"""
    print("[probe-8] 950 原生指令开关宏探测：")
    # A) todo 原文：ccec -dM -E - < /dev/null --soc-version=Ascend950PR
    if tools.get("ccec"):
        rc, out, err = capture_cmd([tools["ccec"], "-dM", "-E", "-", "--soc-version=Ascend950PR"],
                                   stdin_devnull=True)
        log = rep.outdir / "probe8_ccec_todo_option.log"
        log.write_text(f"# rc={rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n", encoding="utf-8")
        rep.add("ccec_todo_command (--soc-version=Ascend950PR)",
                "命中" if rc == 0 else "未命中",
                [rep.evidence(log, 1, f"rc={rc}"), rep.evidence(log, 3, err.splitlines()[0] if err else "")],
                note="todo 原文命令如实执行留账；本 CANN 版本若拒绝该选项则以 stderr 为证")
    # B) ccec 原生拼写兜底：--cce-soc-version（真实选项名）+ --cce-soc-core-type
    if tools.get("ccec"):
        rc, out, err = capture_cmd(
            [tools["ccec"], "-dM", "-E", "-",
             "--cce-soc-version=Ascend950PR", "--cce-soc-core-type=AICore"],
            stdin_devnull=True)
        log = rep.outdir / "probe8_ccec_native_option.log"
        log.write_text(f"# rc={rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n", encoding="utf-8")
        rep.add("ccec_native_option (--cce-soc-version/--cce-soc-core-type)",
                "命中" if rc == 0 else "未命中",
                [rep.evidence(log, 1, f"rc={rc}"), rep.evidence(log, 3, err.splitlines()[0] if err else "")],
                note="ccec 真实选项名拼写兜底（--soc-version 并非本版 ccec 选项）")
    # C) 工具链实际使用的设备模式驱动：bisheng --npu-arch=dav-3510 -xasc -dM -E
    if tools.get("bisheng"):
        rc, out, err = capture_cmd(
            [tools["bisheng"], "--npu-arch=dav-3510", "-xasc", "-dM", "-E", "-"], stdin_devnull=True)
        macro_log = rep.outdir / "probe8_predefined_macros.txt"
        macro_log.write_text(out if out else "", encoding="utf-8")
        log = rep.outdir / "probe8_bisheng_dM.log"
        log.write_text(f"# rc={rc}\n--- stderr ---\n{err}\n", encoding="utf-8")
        if rc == 0:
            defines = [ln for ln in out.splitlines() if ln.startswith("#define")]
            evidence = [rep.evidence(macro_log, 1, f"{len(defines)} predefined macros")]
            for key in ("__CCE__", "__CCE_AICORE__", "__CCE_ARCH__", "__fbuf__"):
                for idx, ln in enumerate(defines):
                    if ln.startswith(f"#define {key} "):
                        evidence.append(rep.evidence(macro_log, idx + 1, ln))
                        break
            rep.add("bisheng_device_mode_macros (--npu-arch=dav-3510 -xasc -dM -E)",
                    "命中", evidence,
                    note=f"完整宏表 {len(defines)} 条落 {macro_log.name}；"
                         "950 原生指令开关以 __CCE_AICORE__ / __fbuf__（cce_fixpipe_buff）等为证")
        else:
            rep.add("bisheng_device_mode_macros", "未命中",
                    [rep.evidence(log, 1, f"rc={rc}")], note="bisheng 设备模式宏探测失败")


def _probe8_object_disasm(rep: Probe8Report, tools: Dict[str, Optional[str]],
                          args: argparse.Namespace) -> Optional[Path]:
    """对编译出的 .o 执行 llvm-objdump -d 自动核验；抽取 .aicore_binary 设备 ELF 再核验。"""
    print("[probe-8] llvm-objdump 反汇编自动核验：")
    objdump = tools.get("llvm-objdump")
    objcopy = tools.get("llvm-objcopy")
    obj = Path(args.probe8_object)
    if not obj.is_absolute():
        obj = OP_DIR / obj
    if not obj.is_file() and tools.get("bisheng") and (OP_DIR / "run.sh").is_file():
        print("[probe-8] 设备对象缺失，自动执行 `bash run.sh kernel` 出包（run.sh 同款容器入口）...")
        rc = run_streaming(["bash", "run.sh", "kernel"], OP_DIR,
                           rep.outdir / "probe8_autobuild_kernel.log")
        if rc != 0:
            rep.add("object_available", "未命中",
                    [rep.evidence(rep.outdir / "probe8_autobuild_kernel.log", 1, f"run.sh kernel rc={rc}")],
                    note="自动出包失败，详见日志")
            return None
    if not obj.is_file():
        rep.add("object_available", "未命中", [],
                note=f"{obj} 不存在且无 bisheng 可自动出包；请先 bash run.sh kernel")
        return None
    rep.add("object_available", "命中", [rep.evidence(obj, 0, str(obj))],
            note=f"size={obj.stat().st_size} bytes")

    if objdump is None:
        rep.add("llvm_objdump_disasm", "未命中", [], note="llvm-objdump 不可用")
        return obj

    rc, out, err = capture_cmd([objdump, "-d", str(obj)])
    host_dis = rep.outdir / "probe8_host_disasm.txt"
    host_dis.write_text(out + err, encoding="utf-8")
    lines = out.splitlines()
    decoded = [(i, ln) for i, ln in enumerate(lines, start=1)
               if re.match(r"^\s+[0-9a-f]+:", ln) and "<not available>" not in ln]
    unavailable = [ln for ln in lines if "<not available>" in ln]
    evidence = [rep.evidence(host_dis, decoded[i][0], decoded[i][1]) for i in range(min(3, len(decoded)))]
    rep.add("host_stub_disasm", "命中" if decoded else "未命中", evidence,
            note=f"宿主侧可解码指令 {len(decoded)} 条；.o 内嵌设备 ELF 待核验")
    return obj


def _probe8_device_elf(rep: Probe8Report, tools: Dict[str, Optional[str]], obj: Optional[Path]) -> Optional[Path]:
    objcopy = tools.get("llvm-objcopy")
    objdump = tools.get("llvm-objdump")
    if obj is None or objcopy is None or objdump is None:
        return None
    rc, _, err = capture_cmd([objcopy, "-O", "binary", "--only-section=.aicore_binary",
                              str(obj), str(rep.outdir / "probe8_aicore.elf")])
    elf = rep.outdir / "probe8_aicore.elf"
    if rc != 0 or not elf.is_file():
        rep.add("device_elf_extract", "未命中",
                [rep.evidence(rep.outdir / "probe8_host_disasm.txt", 1, err.strip()[:200] if err else "")],
                note="无法抽取 .aicore_binary 设备 ELF 段")
        return None
    rc, out, err = capture_cmd([objdump, "-d", str(elf)])
    dev_dis = rep.outdir / "probe8_device_disasm.txt"
    dev_dis.write_text(out + err, encoding="utf-8")
    labels = [(i, ln) for i, ln in enumerate(out.splitlines(), start=1) if re.match(r"^[0-9a-f]+ <.+>:$", ln)]
    na_count = out.count("<not available>")
    evidence = [rep.evidence(dev_dis, labels[i][0], labels[i][1]) for i in range(min(2, len(labels)))]
    evidence.append(rep.evidence(elf, 0, f"size={elf.stat().st_size} bytes"))
    rep.add("device_elf_extract", "命中", evidence,
            note=f"函数符号 {len(labels)} 个；llvm-objdump 解码 <not available> 指令 {na_count} 条")
    return dev_dis if rc == 0 else None


def _probe8_instruction_verdicts(rep: Probe8Report, dev_dis: Optional[Path]) -> None:
    """NDDMA / FixPipe / BufferID 指令记号裁决：能解码才判命中，否则如实「未命中」+ 候选清单。"""
    print("[probe-8] NDDMA / FixPipe / BufferID 指令记号核验：")
    decodable = False
    if dev_dis is not None and dev_dis.is_file():
        body = dev_dis.read_text(encoding="utf-8", errors="replace")
        decodable = bool(re.search(r"^\s+[0-9a-f]+: (?!<not available>)", body, re.M))
    for key, label in (("nddma", "NDDMA 硬件排布指令"), ("fixpipe", "FixPipe 随路指令"),
                       ("bufferid", "BufferID 同步指令")):
        if decodable and dev_dis is not None:
            hits = grep_evidence(dev_dis, [rf"\b{re.escape(c)}" for c in PROBE8_CANDIDATES[key]])
            rep.add(f"instruction:{key}", "命中" if hits else "未命中", hits,
                    candidates=PROBE8_CANDIDATES[key])
        else:
            rep.add(f"instruction:{key}", "未命中", [],
                    note="本 CANN (9.2.0-beta.1) llvm-objdump 注册了 hiipu64 对象格式但无 CCE ISA "
                         "指令解码器，设备指令全部显示 <not available>，反汇编层无法定位、不编造命中",
                    candidates=PROBE8_CANDIDATES[key])


def _probe8_intrinsic_scan(rep: Probe8Report, tools: Dict[str, Optional[str]]) -> None:
    """950 原生指令开关的内置函数/宏层证据：save-temps 预处理体中检索工具链暴露的本征接口。"""
    print("[probe-8] 950 原生指令开关（内置函数/宏层）扫描：")
    bisheng = tools.get("bisheng")
    kernel_src = OP_DIR / "op_kernel" / "sageattention2.cpp"
    if bisheng is None or not kernel_src.is_file():
        rep.add("intrinsic_switch_scan", "跳过", [],
                note="无 bisheng 或缺设备源码，仅在 BiSheng 容器内执行本子步")
        return
    catlass_inc = Path(os.environ.get("CATLASS_ROOT", str(OP_DIR / "docker" / "third_party" / "catlass"))) / "include"
    ascend_inc = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/current") + "/include"
    st_dir = rep.outdir / "save_temps"
    st_dir.mkdir(parents=True, exist_ok=True)
    cmd = [bisheng, "--npu-arch=dav-3510", "-std=c++17", "-xasc", "--cce-aicore-only",
           "-DCATLASS_ARCH=3510", "-DENABLE_CV_COMM_VIA_SSBUF=1", "--cce-no-dcache-flush",
           "-save-temps", "-c", str(kernel_src),
           "-I", ascend_inc,
           "-I", str(OP_DIR / "sageattn_fp8" / "ascendc"),
           "-I", str(OP_DIR / "sageattn_fp8" / "catlass"),
           "-I", str(OP_DIR / "quant"),
           "-I", str(catlass_inc),
           "-o", str(st_dir / "kern.o")]
    # 关键：save-temps 的中间 .asci/.o 以源文件名落在**进程 cwd**，
    # 故 cwd 必须是 st_dir，避免把 18MB 级中间体散落进算子目录
    rc = run_streaming(cmd, st_dir, rep.outdir / "probe8_save_temps_compile.log")
    if rc != 0:
        rep.add("intrinsic_switch_scan", "未命中",
                [rep.evidence(rep.outdir / "probe8_save_temps_compile.log", 1, f"rc={rc}")],
                note="save-temps 设备编译失败")
        return
    for asci in sorted(st_dir.glob("*.asci")):
        core = "cube" if "cube" in asci.name else ("vec" if "vec" in asci.name else "other")
        for key, patterns in PROBE8_INTRINSIC_PATTERNS.items():
            hits = grep_evidence(asci, patterns, max_hits=3)
            if hits:
                rep.add(f"intrinsic_switch:{key}:{core}", "命中", hits,
                        note=f"工具链暴露的 950 指令开关证据（{asci.name}）")


def run_probe8(args: argparse.Namespace) -> int:
    print(f"[probe-8] {PROBE_TITLES[8]}")
    outdir = probe_build_root(args) / "msprof" / args.case / "probe8"
    outdir.mkdir(parents=True, exist_ok=True)
    rep = Probe8Report(outdir)
    tools = _probe8_tool_inventory(rep)
    if not any(tools.get(n) for n in ("ccec", "bisheng", "llvm-objdump")):
        print("[probe-8] 本环境不可用：ccec/bisheng/llvm-objdump 均未找到；请在 BiSheng 容器内执行：\n"
              "    bash docker/sage run python3 benchmarks/profile_single_op.py --probe 8")
        return PROBE_EXIT_UNAVAILABLE

    _probe8_macro_probes(rep, tools)
    obj = _probe8_object_disasm(rep, tools, args)
    dev_dis = _probe8_device_elf(rep, tools, obj)
    _probe8_instruction_verdicts(rep, dev_dis)
    _probe8_intrinsic_scan(rep, tools)

    meta = {
        "case": args.case,
        "soc_todo": "Ascend950PR",
        "arch_real": "dav-3510 (内部 dav-c310 cube/vec)",
        "cann": os.environ.get("ASCEND_HOME_PATH", ""),
        "tools": {k: v for k, v in tools.items() if v},
    }
    report = rep.save(meta)
    print(f"[probe-8] 结构化结论已落 {report}（verdict 只含 命中/未命中/跳过，绝不编造）")
    return 0


PROBE_RUNNERS: Dict[int, Callable[[argparse.Namespace], int]] = {
    1: lambda a: run_msprof_metric_probe(1, a),
    2: lambda a: run_msprof_metric_probe(2, a),
    3: lambda a: run_msprof_metric_probe(3, a),
    4: lambda a: run_msprof_metric_probe(4, a),
    5: run_advisor_probe,
    6: run_npusim_probe,
    7: run_sanitizer_probe,
    8: run_probe8,
}


def run_probes(args: argparse.Namespace) -> int:
    """探针调度：单探针回传其退出码；all 顺序执行全 8 探针，任一环境不可用以 3 收尾。"""
    if args.probe == "all":
        probes = list(range(1, 9))
    else:
        probes = [int(args.probe)]
        if any(no < 1 or no > 8 for no in probes):
            print(f"[profile_single_op] 非法 --probe 取值：{args.probe}（应为 1..8 或 all）。")
            return 2
    results: List[Tuple[int, int]] = []
    for no in probes:
        try:
            rc = PROBE_RUNNERS[no](args)
        except KeyError as err:
            print(f"[probe-{no}] 用例名非法：{err}（合法用例表见 benchmarks/cases.py）。")
            return 2
        results.append((no, rc))
    if len(results) == 1:
        return results[0][1]
    print("[probe-all] 汇总：")
    for no, rc in results:
        state = "SKIP(需真机)" if rc == PROBE_EXIT_UNAVAILABLE else ("OK" if rc == 0 else f"TOOL-FAIL(rc={rc})")
        print(f"  probe-{no}: {state}")
    rcs = [rc for _, rc in results]
    if PROBE_EXIT_UNAVAILABLE in rcs:
        return PROBE_EXIT_UNAVAILABLE
    return max(rcs) if rcs else 0


# ==================================================================================
# 默认路径：单用例 torch_npu.profiler 采集（与探针面无关，保持既有 CLI 完全向后兼容）
# ==================================================================================

def main() -> int:
    args = build_arg_parser().parse_args()

    if args.probe:
        try:
            int(args.probe)
        except ValueError:
            if args.probe != "all":
                print(f"[profile_single_op] 非法 --probe 取值：{args.probe}（应为 1..8 或 all）。")
                return 2
        return run_probes(args)

    try:
        import torch
        import torch_npu
    except ImportError as err:
        print(f"[profile_single_op] torch/torch_npu unavailable ({err}); skipping (exit 0).")
        return 0

    if not torch.npu.is_available():
        print("[profile_single_op] torch.npu.is_available() is False (no NPU device attached); skipping (exit 0).")
        return 0

    if str(OP_DIR / "python") not in sys.path:
        sys.path.insert(0, str(OP_DIR / "python"))
    from sageattention2 import sageattn_qk_int8_pv_fp8_asc

    case = get_case(args.case)
    q, k, v = build_inputs(case, torch)

    def call():
        return sageattn_qk_int8_pv_fp8_asc(
            q, k, v, tensor_layout="HND", is_causal=case.is_causal, pv_accum_dtype="fp32+fp16"
        )

    for _ in range(args.warmup):
        call()
    torch.npu.synchronize()

    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    with torch_npu.profiler.profile(activities=activities) as prof:
        for _ in range(args.steps):
            call()
        torch.npu.synchronize()

    print(f"[profile_single_op] case={case.name} shape=[{case.batch}, {case.heads}, {case.seq}, {case.dim}] "
          f"kv_heads={case.kv_heads} dtype={case.dtype} causal={case.is_causal} tier={case.tier} "
          f"device={torch.npu.get_device_name()}")
    print(prof.key_averages().table(row_limit=30))

    if not args.no_trace:
        trace = Path(args.trace) if args.trace else OP_DIR / "build" / f"profile_{case.name}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        print(f"[profile_single_op] chrome trace written to {trace}")
    print("[profile_single_op] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
