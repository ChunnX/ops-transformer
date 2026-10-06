#!/usr/bin/env python3
"""死代码扫描器（sageattention2 交付物专用，多工具整合）。

机制与边界（先读再用，误报类别见 FP_*）：
- python        AST 级 unused/duplicate import、unused def/arg（模块内保守判定）
- cpp-include   标准头逐符号探测（死 include；传递可用性需人工复核）
- cpp-func      全树零调用函数（正则法；活类别过滤清单见 FP_CPP_FUNC）
- cpp-const     constexpr/#define 零消费
- dup           6 行归一化重复块（同文件近邻 < 2*W 行的忽略）
- cppcheck      cppcheck --enable=unusedFunction 全程序分析（需安装；注意两点：
                ①编译期选择（#if SAGE_DIAG_STAGE / SAGE_ATTN_DEVICE_COMPILE）未取分支
                内的调用点不可见 → 设备侧函数会假性 never used，逐条对码后再删；
                ②文件集必须含全部消费方，否则反向假阳性）
- tidy          clang-tidy host 可编译子集（deadcode.DeadStores / misc-unused-*）；
                设备侧头文件（__aicore__/__gm__/BiSheng 内建）不在此列，为已知边界

用法：
    python3 tools/deadcode_scan.py --all                 # 全部模式
    python3 tools/deadcode_scan.py python cpp-func       # 指定模式
    python3 tools/deadcode_scan.py cppcheck --cppcheck-bin /path/to/cppcheck
退出码：有发现 = 1，干净 = 0（发现需逐条人工对码，非自动删除）。
"""

import argparse
import ast
import collections
import hashlib
import os
import re
import subprocess
import sys

OP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全量源文件集（git ls-files，排除构建产物）
def source_files(exts):
    out = subprocess.run(["git", "ls-files", "-z", "."], cwd=OP_ROOT,
                         capture_output=True, text=True).stdout.split("\0")
    return [os.path.join(OP_ROOT, f) for f in out if f and f.endswith(exts)
            and "/build/" not in f and "/golden/" not in f and "/third_party/" not in f]

# --- 已归档误报类别（历轮裁定记录，删改前先对码台账） ---
FP_PY_IGNORE_DEF = re.compile(
    r"^(Test|test_|pytest_)"      # pytest 发现面与生命周期钩子（如 pytest_sessionfinish）
    r"|^_timeout_handler$"        # signal handler 签名
    r"|^main$")
FP_PY_SIDE_EFFECT_IMPORT = {"torch_npu"}   # 注册 torch.npu 后端的副作用导入
FP_PY_DUP_IMPORT_OK = {"torch", "torch_npu", "F"}  # 函数级重复导入 = spawn 沙箱纪律
FP_CPP_FUNC = re.compile(      # 零调用扫描的活类别（框架宏注册/外部绑定/局部变量等）
    r"^(OP_TYPE_REGISTER|ADD_TO_LAUNCHER_LIST_AICORE|OP_INPUT|OP_ATTR|L2_DFX_PHASE"
    r"|DFX_IN|DFX_OUT|CommonOpExecutorRun|OP_ADD|REG_OP|IMPL_OP|TilingSage"
    r"|ParseSage|aclrt|aclDestroy|aclFinalize|PYBIND11_MODULE)"
    r"|^(SageAttnTwinRunForward|SageAttnTwinQuantDump)"  # ctypes 绑定孪生入口
    r"|^(setenv|unsetenv|memcmp|fflush|memset|memcpy|printf)$"  # C 标准库调用伪检测
    r"|^(NodeInfoWithAttrs|TestInferShapeContextHolder|TestInferDataTypeContextHolder"
    r"|covered|exec_die|out_bits|lse|q_pad|o_acc|s_row|inv_row)$"  # 测试局部构造声明/变量伪检测
)
FP_TIDY_TU = [                  # host 可编译子集（clang-tidy 目标；设备头不可解析）
    "quant/quant_int8.cpp", "quant/quant_fp8.cpp",
    "python/sageattention2/csrc/attention.cpp",
    "tests/test_tiling_struct.cpp", "tests/test_op_host.cpp", "tests/test_aclnn_host.cpp",
    "tools/sim/sim_main.cpp", "tools/sim/sim_reference.cpp",
    "benchmarks/benchmark_aclnn.cpp"]
TIDY_CHECKS = "-*,clang-analyzer-deadcode.DeadStores,misc-unused-parameters," \
              "misc-unused-using-decls,readability-redundant-declaration"

findings = []

def report(kind, path, ln, detail):
    findings.append((kind, path, ln, detail))
    print(f"{kind:14} {os.path.relpath(path, OP_ROOT)}:{ln}  {detail}")


# ---------------- python ----------------
def scan_python(paths):
    # 跨文件名字引用索引（消除模块级 def 被其他文件消费的假阳性）
    global cross_refs
    cross_refs = collections.Counter()
    file_names = {}
    for path in paths:
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            continue
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        file_names[path] = names
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                cross_refs[n.id] += 1
    # 注意：FunctionDef/ClassDef 的名字不是 ast.Name 节点，定义处不产生计数，
    # 因此无需扣除；cross_refs[nm] == 0 即全树零文本引用。
    for path in paths:
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            report("PY-SYNTAX", path, 0, str(e)); continue
        imported, used, all_exports = {}, set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported.setdefault((a.asname or a.name).split(".")[0], node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue  # 编译指令，非导入
                for a in node.names:
                    if a.name != "*":
                        imported.setdefault(a.asname or a.name, node.lineno)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                all_exports |= {e.value for e in ast.walk(node.value)
                                if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        for nm, ln in imported.items():
            if nm not in used and nm not in all_exports and nm not in ann_names(src):
                if nm in FP_PY_SIDE_EFFECT_IMPORT:
                    continue
                report("PY-UNUSED-IMPORT", path, ln, nm)
        seen = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    nm = (a.asname or a.name).split(".")[0]
                    if nm in FP_PY_DUP_IMPORT_OK:
                        continue  # 函数级 torch/F 导入 = spawn 沙箱/子进程纪律（已归档）
                    if nm in seen and seen[nm] != node.lineno:
                        report("PY-DUP-IMPORT", path, node.lineno, nm)
                    seen.setdefault(nm, node.lineno)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if getattr(node, "decorator_list", None):
                    continue  # 带装饰器 = 框架发现面（pytest.fixture 等）
                nm = node.name
                # 跨文件引用计数：本文件外是否还有其他 py 文件引用该名
                ext_refs = cross_refs.get(nm, 0)
                if ext_refs == 0 and nm not in all_exports \
                        and not FP_PY_IGNORE_DEF.search(nm) \
                        and not re.search(rf"['\"]{nm}['\"]", src):
                    report("PY-UNUSED-DEF", path, node.lineno, nm)

def ann_names(src):
    return set(re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']', src))


# ---------------- cpp: 死 include / 零调用函数 / 死常量 ----------------
HEADER_SYMS = {
    "cstdint": [r"\bint\d+_t\b", r"\buint\d+_t\b"],
    "cstddef": [r"\bsize_t\b", r"\bptrdiff_t\b", r"\boffsetof\b", r"\bnullptr_t\b"],
    "cstring": [r"\b(mem|str)[a-z]+\s*\(", r"\bstd::(mem|str)[a-z]+\b"],
    "cstdio": [r"\b(printf|fprintf|fopen|fclose|fread|fwrite|FILE|perror|fflush)\b"],
    "cstdlib": [r"\b(malloc|free|abort|exit|atoi|getenv|strtoll)\b"],
    "cmath": [r"\b(exp2|log2|sqrt|fabs|pow|fmod|nearbyint)[f]?\s*\(", r"\bstd::(exp2|log2|sqrt|fabs|floor|ceil|round|isfinite|isnan|abs)\b"],
    "algorithm": [r"std::(min|max|fill|copy|sort|clamp|find|accumulate|transform|swap)"],  # 无尾 \b：fill_n 等下划线后缀是真使用
    "vector": [r"\bstd::vector\b"], "string": [r"\bstd::string\b"],
    "memory": [r"\bstd::(unique_ptr|shared_ptr|make_unique|make_shared)\b"],
    "type_traits": [r"\bstd::(is_|enable_if|conditional|decay|remove_)\w*\b", r"\bif constexpr\b"],
    "utility": [r"\bstd::(move|forward|pair|exchange|swap)\b"],
    "limits": [r"\bstd::numeric_limits\b"],
}

def strip_cpp(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r'"(\\.|[^"\\])*"', '""', src)
    src = re.sub(r"'(\\.|[^'\\])'", "''", src)
    return src

def scan_cpp_include(paths):
    for path in paths:
        lines = open(path, errors="replace").read().split("\n")
        body = strip_cpp("\n".join(l for i, l in enumerate(lines) if "#include" not in l))
        for i, ln in enumerate(lines):
            m = re.match(r'\s*#\s*include\s*[<"]([^>"]+)[>"]', ln)
            if not m or m.group(1).split("/")[0].split(".")[0] not in HEADER_SYMS:
                continue
            fam = os.path.basename(m.group(1))
            for pat in HEADER_SYMS.get(fam, []):
                if re.search(pat, body):
                    break
            else:
                report("CPP-DEAD-INCLUDE?", path, i + 1, m.group(1))

def scan_cpp_func(paths):
    corpus = "\n".join(strip_cpp(open(p, errors="replace").read()) for p in paths)
    for path in paths:
        src = strip_cpp(open(path, errors="replace").read())
        for m in re.finditer(
                r'^[ \t]*(?:template\s*<[^>]*>\s*)?(?:inline\s+|static\s+|constexpr\s+'
                r'|__inline__\s+|__aicore__\s+|__simd_vf__\s+|__simd_callee__\s+'
                r'|CATLASS_DEVICE\s+|virtual\s+|explicit\s+)*[\w:<>~*&\s,()]*?\b(\w+)\s*\(',
                src, re.M):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "return", "sizeof", "catch",
                        "defined", "main", "operator") or FP_CPP_FUNC.search(name):
                continue
            # 注意：字符类含 \s，惰性跳转可跨行——限定前缀在匹配体内而非行前缀
            body = m.group(0)
            before = body[: body.rfind(name)]
            if before.rstrip().endswith("::"):
                continue  # 限定调用（Class::name( )）非定义
            if re.search(r"\b(?:std::|auto\b|vector|pair)", before):
                continue  # 局部变量构造声明（std::vector<T> name(...)）非函数定义
            if len(re.findall(rf"\b{re.escape(name)}\s*[(<]", corpus)) <= 1:
                report("CPP-ZERO-CALL?", path, src[:m.start()].count("\n") + 1, name)

def scan_cpp_const(paths):
    corpus = "\n".join(strip_cpp(open(p, errors="replace").read()) for p in paths)
    for path in paths:
        src = corpus_parts.get(path, "")
        for m in re.finditer(r'\bconstexpr\s+(?:const\s+)?[\w:<>\s,()&*]+?\b([A-Za-z_]\w*)\s*=', src):
            nm = m.group(1)
            if len(re.findall(rf"\b{re.escape(nm)}\b", corpus)) <= 1:
                report("CPP-DEAD-CONST?", path, src[:m.start()].count("\n") + 1, nm)

corpus_parts = {}


# ---------------- 重复块 ----------------
def scan_dup(paths, width=6):
    blocks = collections.defaultdict(list)
    for path in paths:
        lines = open(path, errors="replace").read().split("\n")
        norms = [re.sub(r'//.*|".*"|\'[^\']*\'', "", l).strip() for l in lines]
        norms = [re.sub(r"\s+", " ", n) for n in norms]
        for i in range(len(norms) - width + 1):
            win = norms[i:i + width]
            if all(win):
                h = hashlib.md5("\n".join(win).encode()).hexdigest()
                blocks[h].append((path, i + 1))
    reported = set()
    for h, locs in blocks.items():
        if len(locs) < 2:
            continue
        files = sorted(set(p for p, _ in locs))
        if len(files) == 1 and max(l for _, l in locs) - min(l for _, l in locs) < width * 2:
            continue
        key = tuple(sorted(locs))
        if key in reported:
            continue
        reported.add(key)
        detail = ", ".join(f"{os.path.basename(p)}:{l}" for p, l in locs[:4])
        report("DUP-BLOCK", files[0], locs[0][1], f"x{len(locs)}: {detail}")


# ---------------- 外部工具包装 ----------------
def run_cppcheck(bin_path, files):
    if not shutil.which(bin_path) and not os.path.exists(bin_path):
        print(f"SKIP cppcheck: {bin_path} 不可执行（brew install cppcheck）")
        return
    cmd = [bin_path, "--enable=unusedFunction", "--std=c++17",
           "-I", os.path.join(OP_ROOT), "-I", os.path.join(OP_ROOT, "quant"),
           "-I", os.path.join(OP_ROOT, "sageattn_fp8/ascendc"),
           "-I", os.path.join(OP_ROOT, "sageattn_fp8/catlass"),
           "-I", os.path.join(OP_ROOT, "op_kernel")] + files
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=OP_ROOT)
    for ln in r.stdout.split("\n"):
        m = re.match(r"(.+?):(\d+):.*\[unusedFunction\]", ln)
        if m:
            report("CPPCHECK-UNUSED?", m.group(1), int(m.group(2)),
                   "设备分支/编译期选择内调用点在宿主视图不可见——逐条对码后再判（见文件头说明②）")

def run_tidy(tidy_bin):
    if not (os.path.exists(tidy_bin) or shutil.which(tidy_bin)):
        print(f"SKIP tidy: {tidy_bin} 不存在")
        return
    inc = ["-I", OP_ROOT, "-I", os.path.join(OP_ROOT, "quant"),
           "-I", os.path.join(OP_ROOT, "sageattn_fp8/ascendc"),
           "-I", os.path.join(OP_ROOT, "op_kernel")]
    for tu in FP_TIDY_TU:
        r = subprocess.run([tidy_bin, f"--checks={TIDY_CHECKS}", os.path.join(OP_ROOT, tu),
                            "--", "-std=c++17"] + inc, capture_output=True, text=True)
        for ln in r.stdout.split("\n"):
            m = re.match(r"(.+?):(\d+):\d+: warning: (.*)", ln)
            if m and ("DeadStores" in ln or "unused" in ln or "redundant" in ln):
                report("TIDY", m.group(1), int(m.group(2)), m.group(3))


import shutil  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description="sageattention2 死代码扫描器（发现需人工对码，非自动删除）")
    ap.add_argument("modes", nargs="*",
                    default=["python", "cpp-include", "cpp-func", "cpp-const", "dup"],
                    choices=["python", "cpp-include", "cpp-func", "cpp-const",
                             "dup", "cppcheck", "tidy", "all"])
    ap.add_argument("--cppcheck-bin", default="cppcheck")
    ap.add_argument("--tidy-bin", default="/opt/homebrew/opt/llvm/bin/clang-tidy")
    args = ap.parse_args()
    modes = set(args.modes)
    if "all" in modes:
        modes = {"python", "cpp-include", "cpp-func", "cpp-const",
                 "dup", "cppcheck", "tidy"}

    py_files = source_files((".py",))
    cpp_files = source_files((".cpp", ".h", ".hpp"))

    global corpus_parts
    if modes & {"cpp-func", "cpp-const"}:
        corpus_parts = {p: strip_cpp(open(p, errors="replace").read()) for p in cpp_files}

    if "python" in modes:
        scan_python(py_files)
    if "cpp-include" in modes:
        scan_cpp_include(cpp_files)
    if "cpp-func" in modes:
        scan_cpp_func(cpp_files)
    if "cpp-const" in modes:
        scan_cpp_const(cpp_files)
    if "dup" in modes:
        scan_dup(py_files + cpp_files)
    if "cppcheck" in modes:
        run_cppcheck(args.cppcheck_bin, cpp_files)
    if "tidy" in modes:
        run_tidy(args.tidy_bin)

    n = len(findings)
    print(f"--- findings: {n}（'?' 结尾的类别为工具宿主视图，需逐条对码："
          f"设备分支/编译期选择内的调用点不可见）---")
    sys.exit(1 if n else 0)

if __name__ == "__main__":
    main()
