#!/usr/bin/env python3
"""Collect aclnn example coverage from docs/zh/op_api_list.md across four ops repos."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOS = ("ops-cv", "ops-math", "ops-nn", "ops-transformer")
API_LIST_REL = Path("docs/zh/op_api_list.md")

# --exec 支持的 soc；与仓 build.sh --soc 一致
EXEC_SOCS = ("ascend910b", "ascend910_93", "ascend950")

# aclnn 文档「产品支持情况」中 npu 标记 / 产品文案 → soc
SOC_NPU_TAGS: dict[str, frozenset[str]] = {
    "ascend910b": frozenset({"910b"}),
    "ascend910_93": frozenset({"A3"}),
    "ascend950": frozenset({"950"}),
}
SOC_TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    # 排除 Atlas 200I/500 A2
    "ascend910b": re.compile(
        r"Atlas\s+A2\s+(?:训练|推理)系列产品", re.IGNORECASE
    ),
    "ascend910_93": re.compile(
        r"Atlas\s+A3\s+(?:训练|推理)系列产品", re.IGNORECASE
    ),
    "ascend950": re.compile(r"Ascend\s*950", re.IGNORECASE),
}

DEFAULT_EXEC_TIMEOUT_SEC = 300


@dataclass
class ApiEntry:
    repo: str
    api_label: str
    api_names: list[str]
    doc_link: str
    op_dir: str | None
    description: str = ""
    has_example: bool = False
    example_path: str | None = None
    # --exec 模式字段
    soc_supported: bool | None = None
    exec_status: str | None = None  # success|failed|no_example|unsupported|...
    exec_detail: str = ""


@dataclass
class RepoStats:
    repo: str
    entries: list[ApiEntry]
    deprecated: list[ApiEntry] = field(default_factory=list)

    @property
    def in_scope_count(self) -> int:
        return len(self.entries)

    @property
    def with_example_count(self) -> int:
        return sum(1 for e in self.entries if e.has_example)

    @property
    def exec_denom_count(self) -> int:
        return sum(1 for e in self.entries if e.soc_supported)

    @property
    def exec_pass_count(self) -> int:
        return sum(1 for e in self.entries if e.exec_status == "success")


@dataclass
class CoverageData:
    root: Path
    repos: tuple[str, ...]
    per_repo: dict[str, RepoStats]
    all_entries: list[ApiEntry]
    all_deprecated: list[ApiEntry] = field(default_factory=list)
    exec_soc: str | None = None

    @property
    def in_scope_count(self) -> int:
        return len(self.all_entries)

    @property
    def with_example_count(self) -> int:
        return sum(1 for e in self.all_entries if e.has_example)

    @property
    def exec_denom_count(self) -> int:
        return sum(1 for e in self.all_entries if e.soc_supported)

    @property
    def exec_pass_count(self) -> int:
        return sum(1 for e in self.all_entries if e.exec_status == "success")


def op_dir_from_link(link: str) -> str | None:
    """../../math/abs/docs/aclnnXxx.md -> math/abs"""
    parts = Path(link.split("#")[0]).parts
    if "docs" not in parts:
        return None
    idx = parts.index("docs")
    op_rel = [p for p in parts[:idx] if p != ".."]
    return "/".join(op_rel) if op_rel else None


def resolve_doc_path(repo_root: Path, doc_link: str) -> Path | None:
    """Resolve op_api_list.md relative link to absolute aclnn doc path."""
    link = doc_link.split("#")[0].strip()
    if not link:
        return None
    candidate = (repo_root / "docs" / "zh" / link).resolve()
    if candidate.is_file():
        return candidate
    # 兼容偶发写成相对仓根的链接
    alt = (repo_root / link).resolve()
    if alt.is_file():
        return alt
    return None


def op_name_from_dir(op_dir: str) -> str:
    """math/abs -> abs（build.sh --run_example 入参）"""
    return Path(op_dir).name


def parse_soc_support(doc_path: Path, soc: str) -> bool | None:
    """Parse「产品支持情况」for whether *soc* is supported.

    Returns:
      True / False if the product row is found;
      None if the section or the soc row cannot be determined.
    """
    if soc not in SOC_NPU_TAGS:
        raise ValueError(f"unsupported soc for support parse: {soc}")

    try:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    section = _extract_product_support_section(text)
    if not section:
        return None

    npu_tags = SOC_NPU_TAGS[soc]
    term_pat = SOC_TERM_PATTERNS[soc]
    # <!-- npu="950" id1 --> ... - <term>...</term>：支持
    block_re = re.compile(
        r'<!--\s*npu="([^"]+)"[^>]*-->\s*\n'
        r"-\s*<term>([^<]*)</term>\s*：\s*(\S+)",
        re.MULTILINE,
    )
    for npu_raw, term, support in block_re.findall(section):
        tags = {t.strip() for t in npu_raw.split(",") if t.strip()}
        if tags & npu_tags or term_pat.search(term):
            return _is_supported_token(support)

    # 无 npu 注释时的纯列表行
    line_re = re.compile(r"-\s*<term>([^<]*)</term>\s*：\s*(\S+)")
    for term, support in line_re.findall(section):
        if term_pat.search(term):
            return _is_supported_token(support)

    # Markdown 表格：| ...产品... | √ / × |
    for line in section.splitlines():
        if "|" not in line or re.match(r"\|\s*-+", line.strip()):
            continue
        if not term_pat.search(line):
            continue
        if re.search(r"[√✓]|支持", line) and not re.search(
            r"[×✕]|不支持", line
        ):
            return True
        if re.search(r"[×✕]|不支持", line):
            return False

    return None


def _extract_product_support_section(text: str) -> str:
    match = re.search(
        r"^##\s*产品支持情况\s*$", text, flags=re.MULTILINE
    )
    if not match:
        return ""
    start = match.end()
    next_h = re.search(r"^##\s+\S", text[start:], flags=re.MULTILINE)
    end = start + next_h.start() if next_h else len(text)
    return text[start:end]


def _is_supported_token(token: str) -> bool:
    token = token.strip()
    if token.startswith("不支持") or token in {"×", "✕", "x", "X"}:
        return False
    if token.startswith("支持") or token in {"√", "✓"}:
        return True
    return False


def _parse_table_row(line: str, repo: str) -> ApiEntry | None:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if not cells:
        return None

    match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", cells[0])
    if not match:
        return None

    label, link = match.group(1).strip(), match.group(2).strip()
    if "aclnn" not in label:
        return None

    api_names = [
        part.strip()
        for part in label.split("&")
        if part.strip().startswith("aclnn")
    ]
    if not api_names:
        return None

    description = cells[1] if len(cells) > 1 else ""
    description = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

    return ApiEntry(
        repo=repo,
        api_label=label,
        api_names=api_names,
        doc_link=link,
        op_dir=op_dir_from_link(link),
        description=description,
    )


def parse_api_list(
    api_list_path: Path, repo: str
) -> tuple[list[ApiEntry], list[ApiEntry]]:
    """Parse docs/zh/op_api_list.md.

    Returns (in_scope_entries, deprecated_entries).
    - 分母仅取「接口列表」；遇到「## 废弃接口」后停止计入分母
    - 「废弃接口」专节条目单独收集；主列表中与之同名的条目也排除出分母
    """
    text = api_list_path.read_text(encoding="utf-8", errors="ignore")
    main_entries: list[ApiEntry] = []
    deprecated_entries: list[ApiEntry] = []
    seen_header = False
    in_deprecated = False

    for line in text.splitlines():
        if re.match(r"^##\s*废弃接口\s*$", line.strip()):
            in_deprecated = True
            seen_header = False
            continue

        if line.startswith("## ") and in_deprecated:
            break

        if not line.startswith("|"):
            continue

        if ("接口名" in line or "废弃接口" in line) and "说明" in line:
            seen_header = True
            continue
        if re.match(r"\|\s*-+", line):
            continue
        if not seen_header:
            continue

        entry = _parse_table_row(line, repo)
        if entry is None:
            continue

        if in_deprecated:
            deprecated_entries.append(entry)
        else:
            main_entries.append(entry)

    deprecated_labels = {e.api_label for e in deprecated_entries}
    deprecated_names = {name for e in deprecated_entries for name in e.api_names}

    in_scope: list[ApiEntry] = []
    extra_deprecated: list[ApiEntry] = []
    seen_dep_labels = set(deprecated_labels)

    for entry in main_entries:
        if entry.api_label in deprecated_labels or any(
            name in deprecated_names for name in entry.api_names
        ):
            if entry.api_label not in seen_dep_labels:
                extra_deprecated.append(entry)
                seen_dep_labels.add(entry.api_label)
            continue
        in_scope.append(entry)

    return in_scope, deprecated_entries + extra_deprecated


def list_example_files(repo_root: Path, op_dir: str) -> list[Path]:
    examples_dir = repo_root / op_dir / "examples"
    if not examples_dir.is_dir():
        return []
    files: list[Path] = []
    for path in examples_dir.rglob("test_aclnn_*.cpp"):
        posix = path.as_posix()
        if "/tests/" in posix:
            continue
        files.append(path)
    return files


def api_token_pattern(api: str) -> re.Pattern[str]:
    """完整标识符匹配：命中 aclnnFoo / aclnnFooGetWorkspaceSize，
    但不命中 aclnnFooV2、aclnnFooTensor 等更长接口名（避免假阳性）。
    """
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(api)}(?:GetWorkspaceSize)?(?![A-Za-z0-9_])"
    )


def match_example(
    repo_root: Path, entry: ApiEntry
) -> tuple[bool, str | None]:
    """覆盖判定：在算子 examples 下对 test_aclnn_*.cpp 检索完整接口名。

    组合接口（aclnnFoo&aclnnInplaceFoo）拆开后，任一接口名命中即算覆盖。
    """
    if not entry.op_dir:
        return False, None

    files = list_example_files(repo_root, entry.op_dir)
    if not files:
        return False, None

    patterns = [(api, api_token_pattern(api)) for api in entry.api_names]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for _api, pattern in patterns:
            if pattern.search(text):
                return True, path.relative_to(repo_root).as_posix()

    return False, None


def pull_repos(root: Path, repos: tuple[str, ...] = REPOS) -> None:
    """git pull latest code for each ops repo under root. Raises on failure."""
    for repo in repos:
        repo_root = root / repo
        if not repo_root.is_dir():
            raise FileNotFoundError(f"repo directory not found: {repo_root}")

        check = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0 or check.stdout.strip() != "true":
            raise RuntimeError(f"not a git repository: {repo_root}")

        print(f"[pull] {repo}: git pull --ff-only ...", file=sys.stderr)
        result = subprocess.run(
            ["git", "-C", str(repo_root), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"git pull failed for {repo}: {detail}")
        summary = (result.stdout or result.stderr or "").strip().splitlines()
        tail = summary[-1] if summary else "ok"
        print(f"[pull] {repo}: {tail}", file=sys.stderr)


def run_repo_example(
    repo_root: Path,
    op_name: str,
    soc: str,
    timeout_sec: int,
) -> tuple[bool, str]:
    """Run `bash build.sh --run_example <op> eager --soc=<soc>` in repo root."""
    build_sh = repo_root / "build.sh"
    if not build_sh.is_file():
        return False, f"missing build.sh under {repo_root}"

    cmd = [
        "bash",
        "build.sh",
        "--run_example",
        op_name,
        "eager",
        f"--soc={soc}",
    ]
    print(f"[exec] start {repo_root.name}/{op_name}: {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        detail = f"timeout after {timeout_sec}s"
        print(f"[exec] FAIL  {repo_root.name}/{op_name}: {detail}", file=sys.stderr)
        return False, detail

    if result.returncode == 0:
        print(f"[exec] OK    {repo_root.name}/{op_name}", file=sys.stderr)
        return True, "ok"

    tail = (result.stderr or result.stdout or "").strip().splitlines()
    detail = " | ".join(tail[-5:]) if tail else f"exit={result.returncode}"
    detail = detail[:500]
    print(f"[exec] FAIL  {repo_root.name}/{op_name}: {detail}", file=sys.stderr)
    return False, detail


def apply_exec_results(
    root: Path,
    data: CoverageData,
    soc: str,
    timeout_sec: int,
) -> None:
    """Fill soc_supported / exec_status for each entry; run build.sh when needed."""
    data.exec_soc = soc
    # (repo, op_name) -> (ok, detail)；同一算子只跑一次
    run_cache: dict[tuple[str, str], tuple[bool, str]] = {}
    pass_n = 0
    fail_n = 0
    skip_n = 0
    no_ex_n = 0

    for entry in data.all_entries:
        repo_root = root / entry.repo
        doc_path = resolve_doc_path(repo_root, entry.doc_link)
        if doc_path is None:
            entry.soc_supported = None
            entry.exec_status = "no_doc"
            entry.exec_detail = "aclnn doc not found"
            skip_n += 1
            print(
                f"[skip] {entry.repo}/{entry.api_label}: no_doc",
                file=sys.stderr,
            )
            continue

        supported = parse_soc_support(doc_path, soc)
        if supported is None:
            entry.soc_supported = None
            entry.exec_status = "support_unknown"
            entry.exec_detail = "product support section missing or soc row not found"
            skip_n += 1
            print(
                f"[skip] {entry.repo}/{entry.api_label}: support_unknown",
                file=sys.stderr,
            )
            continue

        entry.soc_supported = supported
        if not supported:
            entry.exec_status = "unsupported"
            entry.exec_detail = f"doc marks {soc} unsupported; skipped"
            skip_n += 1
            continue

        # 支持该 soc → 计入分母；无 example / 无目录则记失败
        if not entry.op_dir:
            entry.exec_status = "no_op_dir"
            entry.exec_detail = "cannot resolve op_dir from doc link"
            fail_n += 1
            print(
                f"[exec] FAIL  {entry.repo}/{entry.api_label}: no_op_dir",
                file=sys.stderr,
            )
            continue
        if not entry.has_example:
            entry.exec_status = "no_example"
            entry.exec_detail = "no matching test_aclnn_*.cpp"
            no_ex_n += 1
            print(
                f"[exec] FAIL  {entry.repo}/{entry.api_label}: no_example",
                file=sys.stderr,
            )
            continue

        op_name = op_name_from_dir(entry.op_dir)
        cache_key = (entry.repo, op_name)
        first_run = cache_key not in run_cache
        if first_run:
            run_cache[cache_key] = run_repo_example(
                repo_root, op_name, soc, timeout_sec
            )
        ok, detail = run_cache[cache_key]
        entry.exec_status = "success" if ok else "failed"
        entry.exec_detail = detail
        if ok:
            pass_n += 1
        else:
            fail_n += 1
        # 算子级 OK/FAIL 已在 run_repo_example 打印；这里补接口名便于对照
        reused = "" if first_run else " reused"
        print(
            f"[result] {'OK' if ok else 'FAIL'} "
            f"{entry.repo}/{entry.api_label} (op={op_name}{reused})",
            file=sys.stderr,
        )

    print(
        f"[progress] soc={soc} success={pass_n} failed={fail_n} "
        f"no_example={no_ex_n} skipped={skip_n} "
        f"ops_run={len(run_cache)}",
        file=sys.stderr,
    )


def collect_coverage(
    root: Path, repos: tuple[str, ...] = REPOS
) -> CoverageData:
    all_entries: list[ApiEntry] = []
    all_deprecated: list[ApiEntry] = []
    per_repo: dict[str, RepoStats] = {}

    for repo in repos:
        repo_root = root / repo
        api_list = repo_root / API_LIST_REL
        if not api_list.is_file():
            print(f"warning: missing {api_list}", file=sys.stderr)
            continue

        entries, deprecated = parse_api_list(api_list, repo)
        for entry in entries:
            ok, example_path = match_example(repo_root, entry)
            entry.has_example = ok
            entry.example_path = example_path

        per_repo[repo] = RepoStats(
            repo=repo, entries=entries, deprecated=deprecated
        )
        all_entries.extend(entries)
        all_deprecated.extend(deprecated)

    return CoverageData(
        root=root,
        repos=repos,
        per_repo=per_repo,
        all_entries=all_entries,
        all_deprecated=all_deprecated,
    )


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.2f}%"


def coverage_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100, 2)


def build_json_summary(data: CoverageData) -> dict[str, Any]:
    if data.exec_soc:
        return _build_exec_json_summary(data)

    total_in_scope = data.in_scope_count
    total_covered = data.with_example_count
    repos: dict[str, dict[str, Any]] = {}
    for repo in data.repos:
        if repo not in data.per_repo:
            continue
        stats = data.per_repo[repo]
        repos[repo] = {
            "coverage_rate": coverage_rate(
                stats.with_example_count, stats.in_scope_count
            ),
            "in_scope_count": stats.in_scope_count,
            "with_example_count": stats.with_example_count,
        }

    return {
        "total": {
            "coverage_rate": coverage_rate(total_covered, total_in_scope),
            "in_scope_count": total_in_scope,
            "with_example_count": total_covered,
        },
        "repos": repos,
    }


def _build_exec_json_summary(data: CoverageData) -> dict[str, Any]:
    denom = data.exec_denom_count
    passed = data.exec_pass_count
    repos: dict[str, dict[str, Any]] = {}
    for repo in data.repos:
        if repo not in data.per_repo:
            continue
        stats = data.per_repo[repo]
        repos[repo] = {
            "coverage_rate": coverage_rate(
                stats.exec_pass_count, stats.exec_denom_count
            ),
            "in_scope_count": stats.exec_denom_count,
            "with_example_count": stats.exec_pass_count,
            "supported_count": stats.exec_denom_count,
            "pass_count": stats.exec_pass_count,
            "api_list_count": stats.in_scope_count,
        }

    skipped = sum(
        1 for e in data.all_entries if e.exec_status == "unsupported"
    )
    unknown = sum(
        1
        for e in data.all_entries
        if e.exec_status in {"support_unknown", "no_doc"}
    )
    return {
        "soc": data.exec_soc,
        "total": {
            "coverage_rate": coverage_rate(passed, denom),
            "in_scope_count": denom,
            "with_example_count": passed,
            "supported_count": denom,
            "pass_count": passed,
            "api_list_count": data.in_scope_count,
            "unsupported_skipped": skipped,
            "support_unknown": unknown,
        },
        "repos": repos,
    }


def build_report(data: CoverageData) -> str:
    if data.exec_soc:
        return _build_exec_report(data)

    total_in_scope = data.in_scope_count
    total_covered = data.with_example_count
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    repo_label = "、".join(data.repos)

    lines: list[str] = [
        "# ACLNN API Example 覆盖率统计报告",
        "",
        f"生成时间：{now}",
        f"统计根目录：`{data.root}`",
        f"统计仓库：{repo_label}",
        "",
        "## 统计口径",
        "",
        "- **分母**：各仓 `docs/zh/op_api_list.md`「接口列表」表格中的接口条目（按行计；"
        "`aclnnFoo&aclnnInplaceFoo` 计为 1 条）",
        "- **不计入分母**：`## 废弃接口` 专节中的接口；主列表中与废弃专节同名的条目一并排除",
        "- **分子**：在对应算子目录 `examples/**/test_aclnn_*.cpp` 中出现完整接口标识符"
        "（`aclnnFoo` 或 `aclnnFooGetWorkspaceSize`；不把 `aclnnFooV2` 算作 `aclnnFoo`；"
        "`aclnnFoo&aclnnInplaceFoo` 拆开后任一命中即可）",
        "- **算子目录**：由接口文档链接解析（`.../<op_dir>/docs/aclnnXxx.md`）",
        f"- **仓库范围**：{repo_label}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 纳入统计接口总数 | {total_in_scope} |",
        f"| 已有 example 接口数 | {total_covered} |",
        f"| 排除的废弃接口数 | {len(data.all_deprecated)} |",
        f"| **总覆盖率** | **{pct(total_covered, total_in_scope)}** "
        f"({total_covered}/{total_in_scope}) |",
        "",
        "## 各仓覆盖率",
        "",
        "| 仓库 | 纳入统计 | 已有 example | 废弃排除 | 覆盖率 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for repo in data.repos:
        if repo not in data.per_repo:
            continue
        stats = data.per_repo[repo]
        lines.append(
            f"| {repo} | {stats.in_scope_count} | {stats.with_example_count} | "
            f"{len(stats.deprecated)} | "
            f"{pct(stats.with_example_count, stats.in_scope_count)} |"
        )

    lines.extend(
        [
            "",
            "## 接口 example 覆盖情况",
            "",
            "| 仓库 | 接口名 | 算子目录 | 是否有 example | example 路径 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for entry in sorted(data.all_entries, key=lambda x: (x.repo, x.api_label)):
        covered = "是" if entry.has_example else "否"
        op_dir = entry.op_dir or "-"
        example = entry.example_path or "-"
        lines.append(
            f"| {entry.repo} | {entry.api_label} | {op_dir} | {covered} | {example} |"
        )

    uncovered = [e for e in data.all_entries if not e.has_example]
    lines.extend(
        [
            "",
            "## 未覆盖接口列表",
            "",
            f"共 {len(uncovered)} 条。",
            "",
            "| 仓库 | 接口名 | 算子目录 |",
            "| --- | --- | --- |",
        ]
    )
    for entry in sorted(uncovered, key=lambda x: (x.repo, x.api_label)):
        lines.append(
            f"| {entry.repo} | {entry.api_label} | {entry.op_dir or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 不计入分母的废弃接口",
            "",
            f"共 {len(data.all_deprecated)} 条（来自 `## 废弃接口` 专节及主列表中的同名重叠）。",
            "",
            "| 仓库 | 接口名 | 算子目录 |",
            "| --- | --- | --- |",
        ]
    )
    for entry in sorted(data.all_deprecated, key=lambda x: (x.repo, x.api_label)):
        lines.append(
            f"| {entry.repo} | {entry.api_label} | {entry.op_dir or '-'} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_exec_report(data: CoverageData) -> str:
    soc = data.exec_soc or ""
    denom = data.exec_denom_count
    passed = data.exec_pass_count
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    repo_label = "、".join(data.repos)

    lines: list[str] = [
        f"# ACLNN API Example 运行覆盖率报告（{soc}）",
        "",
        f"生成时间：{now}",
        f"统计根目录：`{data.root}`",
        f"目标 soc：`{soc}`",
        f"统计仓库：{repo_label}",
        "",
        "## 统计口径",
        "",
        f"- **分母**：`op_api_list.md` 接口中，对应 aclnn 文档「产品支持情况」标明支持 `{soc}` 的条目",
        f"- **跳过（不计分母）**：文档标明不支持 `{soc}` 的接口",
        "- **分子**：分母内调用该仓 `bash build.sh --run_example <op> eager --soc=<soc>` 执行成功的接口",
        "- **无 example / 文档缺失 / 支持情况不明**：仍尝试归类；其中「支持但无 example」计入分母且记为未通过",
        "- **同一算子目录**只执行一次 build.sh，结果复用到该目录下命中 example 的接口",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 接口列表总数 | {data.in_scope_count} |",
        f"| 支持 {soc}（分母） | {denom} |",
        f"| 执行成功（分子） | {passed} |",
        f"| **运行通过率** | **{pct(passed, denom)}** ({passed}/{denom}) |",
        "",
        "## 各仓运行通过率",
        "",
        "| 仓库 | 接口列表 | 支持(分母) | 成功(分子) | 通过率 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for repo in data.repos:
        if repo not in data.per_repo:
            continue
        stats = data.per_repo[repo]
        lines.append(
            f"| {repo} | {stats.in_scope_count} | {stats.exec_denom_count} | "
            f"{stats.exec_pass_count} | "
            f"{pct(stats.exec_pass_count, stats.exec_denom_count)} |"
        )

    lines.extend(
        [
            "",
            "## 接口执行明细",
            "",
            "| 仓库 | 接口名 | 算子目录 | 支持 | 有 example | 状态 | 详情 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in sorted(data.all_entries, key=lambda x: (x.repo, x.api_label)):
        if entry.soc_supported is True:
            support = "是"
        elif entry.soc_supported is False:
            support = "否"
        else:
            support = "未知"
        has_ex = "是" if entry.has_example else "否"
        status = entry.exec_status or "-"
        detail = (entry.exec_detail or "-").replace("|", "\\|")
        lines.append(
            f"| {entry.repo} | {entry.api_label} | {entry.op_dir or '-'} | "
            f"{support} | {has_ex} | {status} | {detail} |"
        )

    failed = [
        e
        for e in data.all_entries
        if e.soc_supported
        and e.exec_status
        and e.exec_status != "success"
    ]
    lines.extend(
        [
            "",
            "## 支持但未通过列表",
            "",
            f"共 {len(failed)} 条。",
            "",
            "| 仓库 | 接口名 | 算子目录 | 状态 | 详情 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in sorted(failed, key=lambda x: (x.repo, x.api_label)):
        detail = (entry.exec_detail or "-").replace("|", "\\|")
        lines.append(
            f"| {entry.repo} | {entry.api_label} | {entry.op_dir or '-'} | "
            f"{entry.exec_status} | {detail} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_repos_arg(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--repos 不能为空")
    unknown = [p for p in parts if p not in REPOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"未知仓库: {', '.join(unknown)}；可选: {', '.join(REPOS)}"
        )
    # 去重且保持 REPOS 顺序
    selected = tuple(r for r in REPOS if r in set(parts))
    return selected


def _looks_like_stats_root(path: Path) -> bool:
    """True if path contains at least one ops-* repo with op_api_list.md."""
    for repo in REPOS:
        if (path / repo / API_LIST_REL).is_file():
            return True
    return False


def default_stats_root(script_file: Path | None = None) -> Path:
    """Resolve default --root for both skill-tree and flat script layouts.

    - Skill path: ``<root>/.cursor/skills/.../scripts/this.py`` → ``<root>``
    - Flat copy: ``/workspace/example_coverage/this.py`` → walk up / cwd
    """
    script = (script_file or Path(__file__)).resolve()
    parts = script.parts
    # .../<root>/.cursor/skills/<name>/scripts/<file>.py
    if (
        len(parts) >= 6
        and parts[-2] == "scripts"
        and parts[-4] == "skills"
        and parts[-5] == ".cursor"
    ):
        candidate = script.parents[4]
        if _looks_like_stats_root(candidate):
            return candidate

    search_from = [script.parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in search_from:
        cur = start
        for _ in range(8):
            if cur in seen:
                break
            seen.add(cur)
            if _looks_like_stats_root(cur):
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent

    return Path.cwd().resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    default_root = default_stats_root()
    parser = argparse.ArgumentParser(
        prog="collect_aclnn_example_coverage.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "统计 ops-cv / ops-math / ops-nn / ops-transformer 四仓的 "
            "aclnn API example 覆盖率。\n"
            "静态模式分母：各仓 docs/zh/op_api_list.md「接口列表」；\n"
            "静态模式分子：examples/**/test_aclnn_*.cpp 中出现完整接口标识符。\n"
            "--exec <soc>：按文档平台支持过滤后调用 build.sh 跑 example，"
            "支持=分母、跑通=分子。"
        ),
        epilog=(
            "示例:\n"
            "  # 生成 Markdown 全量报告（默认四仓）\n"
            "  python3 collect_aclnn_example_coverage.py --root /path/to/example_Statistics\n"
            "\n"
            "  # 只统计部分仓\n"
            "  python3 collect_aclnn_example_coverage.py --root /path/to/example_Statistics "
            "--repos ops-math,ops-nn\n"
            "\n"
            "  # 按 soc 执行 example（需本机 CANN/设备就绪）\n"
            "  python3 collect_aclnn_example_coverage.py --root /path/to/example_Statistics "
            "--exec ascend950 --repos ops-cv\n"
            "\n"
            "  # 统计前 git pull，并同时产出 Markdown + JSON\n"
            "  python3 collect_aclnn_example_coverage.py --root /path/to/example_Statistics "
            "--format both --pull\n"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        metavar="DIR",
        help=(
            "包含四仓（ops-cv、ops-math、ops-nn、ops-transformer）的父目录 "
            f"(默认: {default_root})"
        ),
    )
    parser.add_argument(
        "--repos",
        type=parse_repos_arg,
        default=REPOS,
        metavar="LIST",
        help=(
            "参与统计的仓库，逗号分隔；可选 "
            f"{','.join(REPOS)}（默认: 全部）"
        ),
    )
    parser.add_argument(
        "--exec",
        dest="exec_soc",
        choices=EXEC_SOCS,
        default=None,
        metavar="SOC",
        help=(
            "启用运行模式：解析 aclnn 文档平台支持后，对支持的接口调用 "
            "build.sh --run_example；可选 "
            + ", ".join(EXEC_SOCS)
        ),
    )
    parser.add_argument(
        "--exec-timeout",
        type=int,
        default=DEFAULT_EXEC_TIMEOUT_SEC,
        metavar="SEC",
        help=f"单个算子 build.sh 超时秒数（默认: {DEFAULT_EXEC_TIMEOUT_SEC}）",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="markdown",
        metavar="FMT",
        help=(
            "输出格式: markdown=全量 Markdown 报告(默认); "
            "json=仅摘要 JSON; both=同时产出两者"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Markdown 报告路径 "
            "(默认: <root>/aclnn_example_coverage_report.md；"
            "--exec 时为 <root>/aclnn_example_exec_<soc>_report.md)"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "JSON 摘要路径；--format json 且未指定时打印到 stdout；"
            "--format both 且未指定时写默认路径"
        ),
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="统计前对选定仓分别执行 git pull --ff-only；失败则非 0 退出",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    repos: tuple[str, ...] = args.repos

    if args.pull:
        try:
            pull_repos(root, repos=repos)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    data = collect_coverage(root, repos=repos)
    if args.exec_soc:
        apply_exec_results(
            root, data, args.exec_soc, timeout_sec=args.exec_timeout
        )

    default_md = (
        root / f"aclnn_example_exec_{args.exec_soc}_report.md"
        if args.exec_soc
        else root / "aclnn_example_coverage_report.md"
    )
    default_json = (
        root / f"aclnn_example_exec_{args.exec_soc}_summary.json"
        if args.exec_soc
        else root / "aclnn_example_coverage_summary.json"
    )

    if args.format in ("markdown", "both"):
        md_output = args.output or default_md
        md_output.write_text(build_report(data), encoding="utf-8")
        print(
            f"Report written to {md_output}",
            file=sys.stderr if args.format == "both" else sys.stdout,
        )

    if args.format in ("json", "both"):
        payload = build_json_summary(data)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.json_output is not None:
            args.json_output.write_text(text, encoding="utf-8")
            print(f"JSON written to {args.json_output}", file=sys.stderr)
        elif args.format == "both":
            json_path = default_json
            json_path.write_text(text, encoding="utf-8")
            print(f"JSON written to {json_path}", file=sys.stderr)
        else:
            sys.stdout.write(text)


if __name__ == "__main__":
    main()
