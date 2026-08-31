# 代码检视报告
**项目名称**：moe_ep dump 分析工具检视报告
**检视模块**：mc2/tools/dump_analysis/dump_analysis_moe_ep.py, dump_analysis.sh
**检视人**：Turing Team
**检视日期**：2026-08-31


## 🔍 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 3 个 |
| 严重级（CRITICAL）问题 | 1 个 |
| 中等级（MEDIUM）问题 | 1 个 |
| 轻微级（LOW）问题 | 1 个 |
| 误报数量 | 0 个 |
| 已修复问题 | 2 个 |
| 存疑问题 | 1 个 |

**核心结论**：ISSUE-001（路径遍历）和 ISSUE-002（Shell glob）已修复并验证通过，ISSUE-003 为低风险边界场景存疑。核心 per_core_diag 解析逻辑安全，边界保护完善。

---

## ❌ 问题详情及修改建议

### 问题ID：ISSUE-001 | 严重级别：CRITICAL（严重） | 状态：✅ 已修复

#### 🔬 假设检验过程
**代码段**：dump_analysis_moe_ep.py main() 参数解析 — card_names 拼接路径
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 规范违反 | 5.2 | `card_names` 来自 `sys.argv[4]`（外部输入），直接用于 `os.path.join(target_dir, name)` 拼接路径，未做规范化处理 | +40% | 40% |
| 2 | 上下文防御缺失 | 5.2 | Python 侧无独立的 `basename` 或路径规范化防御，仅依赖 sh 脚本侧 `basename` | +30% | 70% |

**结论**：自信值 **70%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联规范条款**：规则 5.2「使用外部数据构造的文件路径前必须进行校验，校验前必须对文件路径进行规范化处理」
**代码路径**：mc2/tools/dump_analysis/dump_analysis_moe_ep.py:598, 611-612
**问题类型**：路径遍历风险
**问题描述**：`card_names` 来自命令行参数 `sys.argv[4]`，经 `split(",")` 后直接传入 `os.path.join(target_dir, name)` 拼接路径。若用户直接调用 Python 脚本并传入 `card_names="../../etc"`，将产生路径遍历，访问到 `target_dir` 之外的目录。虽然 sh 脚本侧通过 `basename` 做了防御，但 Python 侧缺少独立校验，违反规则 5.2。

#### 修改建议
**修改前代码**：
```python
card_names = sys.argv[4].split(",") if len(sys.argv) > 4 else []
```
**修改后代码**：
```python
card_names = [os.path.basename(name) for name in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
```
**修改说明**：对每个 card_name 调用 `os.path.basename()` 去除路径前缀，防止 `../` 路径遍历。符合规则 5.2 的规范化要求，与 sh 脚本侧 `basename` 防御保持一致。

**验证结果**：传入 `card_names="../../etc,0"`，`../../etc` 被 `basename` 截为 `etc`，访问 `dump_dir/etc`（不存在）报错跳过，卡 1 正常解析。路径遍历已阻止。

---

### 问题ID：ISSUE-002 | 严重级别：MEDIUM（中等） | 状态：✅ 已修复

#### 🔬 假设检验过程
**代码段**：dump_analysis.sh find 命令 -name 参数
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 规范违反 | 7.3 | `find` 命令中 `-name "exception_info"*` 的 `*` 在双引号外，会被 shell 先做 glob 展开 | +40% | 40% |

**结论**：自信值 **40%** < 60%，未推翻原假设，但存在隐患，已修复。

---

**关联规范条款**：规则 7.3「避免在命令解析器中使用通配符」
**代码路径**：mc2/tools/dump_analysis/dump_analysis.sh:210
**问题类型**：Shell glob 意外展开
**问题描述**：`-name "exception_info"*` 中 `*` 在引号外，shell 会先在当前目录尝试展开 `exception_info*`，如果当前目录有匹配文件会干扰 find 的 `-name` pattern。应将 `*` 放入引号内。

#### 修改建议
**修改前代码**：
```bash
if find "$TARGET_DIR" -maxdepth 2 -name "exception_info"* -path "*/[0-9]*/*" | grep -q . 2>/dev/null; then
```
**修改后代码**：
```bash
if find "$TARGET_DIR" -maxdepth 2 -name "exception_info*" -path "*/[0-9]*/*" | grep -q . 2>/dev/null; then
```
**修改说明**：将 `*` 放入双引号内，确保由 find 解释 glob pattern 而非 shell 展开。

**验证结果**：语法检查通过，四种 dump 数据正常解析。

---

### 问题ID：ISSUE-003 | 严重级别：LOW（轻微） | 状态：⚠️ 存疑

#### 🔬 假设检验过程
**代码段**：dump_analysis.sh file_num 计算
**假设**：H0: file_num 不可能为 0

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 数据流追踪 | - | `card_dirs_list` 由 `find` 填充，理论上 `is_multi_card=1` 时至少有一个匹配，但存在竞态条件（目录在两次 find 间被删除）导致 `card_dirs_list` 为空、`file_num=0` 的可能 | +25% | 25% |

**结论**：自信值 **25%** < 60%，未推翻原假设。低风险，存疑。

---

**代码路径**：mc2/tools/dump_analysis/dump_analysis.sh:215
**问题类型**：file_num 边界场景
**问题描述**：第 210 行 `find` 检测到数字子目录下有 `exception_info` 文件时设 `is_multi_card=1`，第 214 行 `find` 收集数字子目录到 `card_dirs_list`。若两次 find 之间目录被删除（竞态），`card_dirs_list` 可能为空，`file_num=0`。此时后续 `for card_dir in "${card_dirs_list[@]}"` 不会遍历，不会崩溃，但 `SHARE_EXPERT_CARD_COUNT -gt 0` 的判断会误报 error。

#### 修改建议
无需修复。`file_num=0` 时脚本行为安全（不遍历、报 error），仅日志信息可能误导。如需加固可在第 215 行后添加：
```bash
if [ "$is_multi_card" -eq 1 ] && [ $file_num -eq 0 ]; then
    echo "error:未找到数字命名的卡目录"
    exit 1
fi
```

---

## ✅ 安全代码段确认

以下代码段经假设检验确认安全，无需修改：

| 代码段 | 文件:行号 | 检视结论 |
|--------|-----------|---------|
| `analyze_per_core` record 数组访问 | dump_analysis_moe_ep.py:266-270 | `max_cores = raw_data.shape[0] // PER_CORE_SLOT_SIZE` 整除保证 slot 完整，`record[4]` 访问安全（64B = 16 个 uint32） |
| `print_per_core` 列表索引 | dump_analysis_moe_ep.py:285-288 | `per_core_data` 长度 = `core_count`（break 时不 append），索引不越界 |
| `card_dirs_list` 遍历 | dump_analysis.sh:256,319,368 | `for card_dir in "${card_dirs_list[@]}"` + `basename` 安全引用，无空格问题 |
| `card_names_str` 生成 | dump_analysis.sh:330 | `basename` + `paste -sd,` 正确生成逗号分隔列表 |

## 报告生成时间
2026-08-31
## 报告状态
ISSUE-001/002 已修复并验证通过，ISSUE-003 存疑待确认
