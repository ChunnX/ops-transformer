# =====================================================================
# engram_test C1/C2/数据生成基准 批量任务配置
#   启动: bash ~/.claude/skills/batch-runner/batch_run.sh start <本文件>
#   状态: bash ~/.claude/skills/batch-runner/batch_run.sh status <本文件>
#   停止: bash ~/.claude/skills/batch-runner/batch_run.sh stop  <本文件>
# 任务顺序: c1(框架修复+语义) → c2(数据生成重构+缓存) → bench(全规格生成基准)
# =====================================================================
WORKDIR="/home/qianlong/workspace/cann/ops-transformer.worktrees/engram-local"
OPENCODE_CONFIG="$HOME/.config/opencode/autogen/batch.json"

TASK_TIMEOUT=3600          # 单次尝试 60 分钟（engram_test.py 全量重写较重）
RETRY_PER_ACCT=2           # 同一账号即时重试次数
RESET_AT="00:05"           # 三账号额度耗尽后休眠至此自动续跑
MAX_FAILCNT=6              # 任务累计非额度失败熔断阈值

# 账号轮换顺序：caiqiang → qianlong → lirunhui（额度耗尽自动切下一个）
MODELS=(
  "caiqiang|cq-provider/GLM-5.3"
  "qianlong|glm-provider/GLM-5.3"
  "lirunhui|qwen-provider/GLM-5.3"
)

TASKS=(c1 c2 bench)

# ---- c1: 框架修复 + total-vocab 语义（设计文档 M1/M4/M6） ----
T_c1_OUT="examples/mc2/engram_test/engram_test.py"
T_c1_PROMPT="examples/mc2/engram_test/batch/prompts/c1_framework.txt"
T_c1_VERIFY='python3 -m py_compile {output} && bash -n examples/mc2/engram_test/run_engram.sh && grep -q -- "--shape_semantics" {output} && grep -q "num_entries_total" {output} && grep -q "sys.exit(0 if all_success else 1)" {output} && grep -q "get(timeout" {output} && grep -q "from cann_ops_transformer import ElasticBuffer" {output} && grep -q "repeated sampling" {output} && [ -f examples/mc2/engram_test/README.md ] && git diff --quiet HEAD -- examples/mc2/engram_test/engram_test.py examples/mc2/engram_test/run_engram.sh && git log -10 --format=%s | grep -q "^C1:"'

# ---- c2: 数据生成重构 + 缓存复用（设计文档 M2/M3） ----
T_c2_OUT="examples/mc2/engram_test/bench_prep.py"
T_c2_PROMPT="examples/mc2/engram_test/batch/prompts/c2_datagen.txt"
T_c2_VERIFY='python3 -m py_compile {output} && python3 -m py_compile examples/mc2/engram_test/prep_cache.py && python3 -m py_compile examples/mc2/engram_test/engram_test.py && bash -n examples/mc2/engram_test/run_engram.sh && grep -q "memmap" examples/mc2/engram_test/prep_cache.py && grep -q "SeedSequence" examples/mc2/engram_test/prep_cache.py && grep -q "GEN_VERSION" examples/mc2/engram_test/prep_cache.py && git diff --quiet HEAD -- examples/mc2/engram_test/engram_test.py examples/mc2/engram_test/run_engram.sh examples/mc2/engram_test/prep_cache.py examples/mc2/engram_test/bench_prep.py && git log -10 --format=%s | grep -q "^C2:" && python3 examples/mc2/engram_test/bench_prep.py --specs functional --report /tmp/c2_verify.md && grep -q "ALL SPECS PASS" /tmp/c2_verify.md'

# ---- bench: 全规格数据生成基准（本机可直接跑，仅 numpy+torch） ----
T_bench_OUT="examples/mc2/engram_test/prep_bench_report.md"
T_bench_CMD='python3 examples/mc2/engram_test/bench_prep.py --specs all --report examples/mc2/engram_test/prep_bench_report.md'
T_bench_VERIFY='grep -q "ALL SPECS PASS" {output}'
