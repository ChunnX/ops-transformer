# MegaMoe GMM2-GM wave 实验交付

本文只覆盖 Ascend 950、MTE topology、generic ND/A8W8-NZ GMM、
`combine_quant_mode=0`。A8W4、A4W4/A4W4-NZ、URMA、layered 和 combine-quant
保持原路径。

## 状态与分支

- `[LOCAL-VERIFIED]` 使用用户指定的 `upstream/master`：实现起点为
  `38854175411a507ded2429e9d6d89d703b41ed9e`；收口期间主线前进到
  `916d211e9a5037fcf56f2d0f308efe70492a12a2`，两条分支均已合入并用其
  ops-tensor `051664f3` 依赖重打完整包。handoff 中的旧 base 仅作背景，没有
  cherry-pick 旧 GMM2-GM/diagnostic 实现。
- `[LOCAL-VERIFIED]` token：`codex/0715-gmm2-gm-rebuild-token`，核心实现 head
  `7c91d3e07`，动态 wave tiling 修复 `8224210e1`。
- `[LOCAL-VERIFIED]` tile：`codex/0715-gmm2-gm-rebuild-tile`，核心实现 head
  `e653e6d59`，动态 wave tiling 修复 `a9e04d725`。
- `[NEEDS-NPU]` 两条分支都尚未做上板 correctness、hang、scalar GM 地址、timeline
  和性能验收；在数据返回前不能称为正确版本或最佳版本。

提交边界如下：

```text
9c617e213 fix(mega_moe): own epilogue init parameters
746e6a0da feat(mega_moe): reserve no-quant GMM2 GM workspace
0936430ef feat(mega_moe): write generic GMM2 output directly to GM
ae3b13467 feat(mega_moe): pipeline token combine by expert wave
b0fc5de4a feat(mega_moe): add wave timeline diagnostics
7c91d3e07 fix(mega_moe): trace GMM1 after dispatch ready

tile only:
ede7ec831 feat(mega_moe): distribute combine by GMM2 tile
25167a2fb feat(mega_moe): add wave timeline diagnostics
e653e6d59 fix(mega_moe): trace GMM1 after dispatch ready

latest upstream integration:
916d211e9 upstream/master dependency/namespace cleanup
75c1a4ef0 token merge and package point
e9116a4a7 tile merge and package point

dynamic wave tiling fix:
8224210e1 token: derive wave size from tile workload and AIC count
a9e04d725 tile: equivalent cherry-pick, retaining tile ownership tests
```

## 流水与资源约束

三条独立时间线是：

```text
AIC : GMM1(B) -> GMM2(B) 256x256 直写 GM -> GMM1(B+1)
AIV0: SwiGLU(B) -> SwiGLU(B+1)
AIV1: 完整 Dispatch(B+1) -> Combine(B) -> 完整 Dispatch(B+2) -> Combine(B+1)
```

AIV0 不进入新 combine、不 poll ready。Dispatch 和 combine 共用 AIV1，因此同一颗
AIV1 上按上述顺序串行；AIV0 SwiGLU、AIC 下一 wave GMM1 可以与 AIV1 combine 并行。
rolling 路径没有新增 batch 级 `PIPE_ALL`/`SyncAll`；只保留函数末尾原有的终态同步和
具体 buffer 的 MTE event。

GMM2 输出为 `maxOutputSize * H * sizeof(bfloat16_t)` 的整 kernel 独占 GM 区域。
expert 的行通过累计 token offset 落在该区域，既不是 batch slot 也不是 ring slot，直到
kernel 结束都不会复用，因此 AIC 发布 ready 后无需等待 consumer ACK。所有 workspace
大小和地址偏移使用 64 位计算；输出区按 512 B 对齐，随后是每 local expert 一条 64 B
cache line 的 ready 区。ready slot 在单次 kernel 内也不复用。

ready 协议为真正的 producer fan-in：每颗已启动 AIC 先 `FIX_S` drain 自己此前的 GM
写，再向共享 slot `AtomicAdd(1)`；AIV1 用 bypass-D-cache read 等到实际 `blockNum`。
默认 per-expert，per-batch 为 A/B 开关。固定 case 中：

- per-expert：每 rank `36 * 16 = 576` 次 publish；
- per-batch：wave 为 `5,5,5,1`，每 rank `36 * 4 = 144` 次 publish。

wave 大小不再由 kernel 编译宏或 `ceil(blockNum/worldSize)` 决定。host tiling 先估计每个
expert 的平均 token 数，再按当前 256x256 GMM tile 和实际 `aicNum` 计算：

```text
expectedM = ceil(bs * topK / expertPerRank)
gmm1Tiles = max(1, ceil(expectedM / 256)) * ceil(hiddenDim / 512)
gmm2Tiles = ceil(expectedM / 256) * ceil(h / 256)
expertsPerBatch = clamp(ceil(aicNum / max(1, min(gmm1Tiles, gmm2Tiles))),
                        1, min(expertPerRank, 14))
```

固定 case 的 `expectedM=256`，GMM1/GMM2 分别为 8/16 tiles/expert，因此 tiling 输出
`ceil(36/min(8,16))=5`；kernel 只消费 `expertsPerBatch`。

token 分支把每个 expert 的 M 行在 36 个 logical AIV1 间做平衡连续切分，M=256 时每颗
约 7/8 行；每行从 GM 搬一次，再按 256 列发送。tile 分支把 256x256 tile 按 row-major
round-robin 分给 logical AIV1；固定 M=256,H=4096 时共 16 tile，16 颗 AIV1 各一块。
每块 BF16 tile 一次 2-D GM-to-UB，路由 triple 一次性搬入最多 `256 * 8 * 4 = 8 KiB`，
再调用一次既有 `CombineTokens`。M=0、尾 M 和不同 N 的 ownership grid 有 host test
覆盖；当前合法 H 是 1024 的倍数，所以 256 列 tile 不会产生尾 N。若以后放宽 host
约束，尾 N 的 2-D copy 数值正确性需要新增测试和 NPU 验收。

## 本地验证结果

工具链：CANN `9.0.0-beta.2`，Ascend 950 完整构建路径。

| 项目 | 结果 |
|---|---|
| upstream/master baseline `--pkg` | `[LOCAL-VERIFIED]` PASS |
| token host UT | `[LOCAL-VERIFIED]` 15/15 |
| tile host UT | `[LOCAL-VERIFIED]` 16/16 |
| token 默认/per-batch opkernel | `[LOCAL-VERIFIED]` 每组 8/8 变体编译 |
| tile 默认 opkernel | `[LOCAL-VERIFIED]` 8/8 变体编译 |
| token 默认 `--pkg` | `[LOCAL-VERIFIED]` PASS，4,040,039 bytes，SHA256 `53c0fe602f39ab4694ad81eb77c6f091e6445c4fa794260ea57dd1b7710751ec` |
| tile 默认 `--pkg` | `[LOCAL-VERIFIED]` PASS，4,039,698 bytes，SHA256 `12663ef786d94cd5518291ce871de687598c9dfbb328851e037310cb213dfa17` |
| NPU correctness/timeline/perf | `[NEEDS-NPU]` |

本地归档包分别为 `/tmp/mega_moe_wave_artifacts/token_8224210e1_default.run` 和
`/tmp/mega_moe_wave_artifacts/tile_a9e04d725_default.run`。`/tmp` 不是长期存储，正式上板
建议在目标机器 checkout 对应 branch 后重编。

复现本地 build：

```bash
bash build.sh --ophost_test --soc=ascend950 --ops=mega_moe -j16 -O3
bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3
```

主要编译开关：

```text
MEGA_MOE_GMM2_READY_MODE=0       per-expert
MEGA_MOE_GMM2_READY_MODE=1       per-batch（默认）
MEGA_MOE_GMM2_GM_READY_SMOKE_TEST=1  最小 scalar GM 地址 smoke
MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG=1  默认关闭的内部时间戳
MEGA_MOE_WEIGHT1_INTERLEAVED=1   GMM1/SwiGLU 双 UB slot interleave（默认）
```

`MEGA_MOE_WEIGHT1_INTERLEAVED=1` 不只是调度开关：kernel 会按参考分支
`codex/0715-prefetch-next-batch-dispatch-ab` 的契约，把 weight1 和 weight1 MX scale
解释为匹配的 `[gate, up]` tile-interleaved 布局。上板时必须同步启用 runner/权重准备侧的
interleave；普通连续 gate/up 权重需要显式设置 `MEGA_MOE_WEIGHT1_INTERLEAVED=0`。

默认最优版本不需要额外编译选项：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3

rg -n '#define MEGA_MOE_(WEIGHT1_INTERLEAVED|GMM2_READY_MODE)' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp \
  mc2/mega_moe/op_kernel/arch35/mega_moe.h
```

该模式下 generic GMM1 的 AIC 与 AIV0/SwiGLU 用两个 128KB UB slot 做 ownership
ping-pong；SwiGLU 完成计数按 256-row wave 发布，GMM2 可以追随已完成 wave。终态只 drain
最后一到两个 outstanding slot。A8W4 GMM1 仍走原 GM-staged epilogue，不启用这套 UB
ping-pong。

带宏构建后必须立刻核实宏确实传入生成配置：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
  --ops-compile-options '-DMEGA_MOE_GMM2_READY_MODE=1'

rg -n 'MEGA_MOE_GMM2_READY_MODE=1' \
  build/autogen/custom_compile_options.ini \
  build/impl/dynamic/mega_moe.py \
  build/binary/ascend950/src/mega_moe_apt/MegaMoe.py

./build_out/cann-ops-transformer-custom_linux-x86_64.run
```

上板时先给两个 worktree 路径赋值，后续命令都显式在对应源码目录构建：

```bash
TOKEN_SRC=/path/to/codex-0715-gmm2-gm-rebuild-token
TILE_SRC=/path/to/codex-0715-gmm2-gm-rebuild-tile
```

## 固定 EP8 case

runner 位于 `/home/mace/workspace/moe_test/mega_moe`。它会自行 fork 8 个 rank，不要再套
`torchrun` 或 `mpirun`。固定配置为全局 4096 token、32768 route、128 expert，平均
M=256：

| 列 | 值 |
|---|---|
| bs, h, n, k | 512, 4096, 4096, 8 |
| graph_type | 0 |
| quant_mode / combine_quant_mode | 4 / 0 |
| quant_mode_type / swiglu_quant_dtype / weight_dtype | float8_e4m3fn |
| weight_scale1 / weight_scale2 | float8_e8m0 |
| ep_world_size | 8 |
| expert_per_rank / local_moe_expert_num | 16 / 16 |
| moe_expert_num / max_recv_token_num | 128 / 32768 |
| clamp_limit / seed | 9999 / 831 |

下面生成两行 xlsx：Excel row 2 做 correctness，row 3 做 performance。当前 runner 会自行
重算 `moe_expert_num` 和 `max_recv_token_num`，表中仍显式保留便于审计。

```bash
cd /home/mace/workspace/moe_test/mega_moe
RUN_ROOT=/tmp/mega_moe_gmm2_wave_ep8
CASE_XLSX=$RUN_ROOT/gmm2_wave_ep8.xlsx
SHEET=EP8_M256_H4096
CASE_NAME=MegaMoe_EP8_BS512_H4096_N4096_K8_E16_SEED831
mkdir -p "$RUN_ROOT"

python3 - "$CASE_XLSX" "$SHEET" "$CASE_NAME" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path, sheet, case_name = sys.argv[1:]
columns = [
    "case_name", "bs", "h", "n", "k", "max_recv_token_num",
    "graph_type", "quant_mode", "combine_quant_mode",
    "quant_mode_type", "swiglu_quant_dtype", "weight_dtype",
    "clamp_limit", "weight_scale1", "weight_scale2",
    "ep_world_size", "expert_per_rank", "local_moe_expert_num",
    "moe_expert_num", "seed", "is_performance",
]
base = dict(
    case_name=case_name, bs=512, h=4096, n=4096, k=8,
    max_recv_token_num=32768, graph_type=0, quant_mode=4,
    combine_quant_mode=0, quant_mode_type="float8_e4m3fn",
    swiglu_quant_dtype="float8_e4m3fn", weight_dtype="float8_e4m3fn",
    clamp_limit=9999, weight_scale1="float8_e8m0",
    weight_scale2="float8_e8m0", ep_world_size=8, expert_per_rank=16,
    local_moe_expert_num=16, moe_expert_num=128, seed=831,
)
rows = [dict(base, is_performance=0), dict(base, is_performance=1)]
Path(path).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=columns).to_excel(path, sheet_name=sheet, index=False)
PY
```

## runner 安全处理

原 runner 在 main 开头无条件删除 `/root/atc_data/*`，且子进程异常会被吞掉、rank 等待
没有 timeout。因此先在同一目录做安全副本；保留目录位置是为了不破坏相对 import。

```bash
RUNNER_SRC=$PWD/batch_run_once/mega_moe_once.py
RUNNER_SAFE=$PWD/batch_run_once/mega_moe_once_gmm2_safe.py
cp "$RUNNER_SRC" "$RUNNER_SAFE"
sed -i '1211c\    # SAFE COPY: disabled unconditional deletion of /root/atc_data/*' "$RUNNER_SAFE"
sed -n '1209,1214p' "$RUNNER_SAFE"

if rg -n 'rm /root/atc_data' "$RUNNER_SAFE"; then
  echo 'unsafe deletion is still present' >&2
  exit 1
fi
python3 -m py_compile "$RUNNER_SAFE"
```

若上板 runner 已变化，不要盲用固定行号；先 `rg -n '/root/atc_data'`，只删除该语句，
然后复查 diff。每次 trial 使用独立、无空格的 `output_dir`，因为 runner 会清除该目录下
旧的 `Profiling_Result_*`。runner 固定使用 TCP 端口 50001，不能并发跑两组；启动前用
`ss -ltnp | rg ':50001\b'` 检查，并只清理由本实验遗留的进程。

生成 golden：

```bash
GOLDEN_OUT=$RUN_ROOT/golden
mkdir -p "$GOLDEN_OUT"
timeout --kill-after=60s 14400s python3 "$RUNNER_SAFE" \
  --excel "$CASE_XLSX" --sheet "$SHEET" \
  --start_line 2 --end_line 3 --run_mode 0 \
  --output_dir "$GOLDEN_OUT" --golden \
  >"$GOLDEN_OUT/run.log" 2>&1
rg -n 'golden已全部生成|--golden, skip NPU' "$GOLDEN_OUT/run.log"
test -f "$PWD/input/$SHEET/$CASE_NAME/x_final.pt"
test -f "$PWD/input/$SHEET/$CASE_NAME/expert_token_nums.pt"
```

## Correctness `[NEEDS-NPU]`

每安装一个新包，修改 `VARIANT` 并使用新目录：

```bash
VARIANT=token_per_expert
CORR_OUT=$RUN_ROOT/$VARIANT/correctness
mkdir -p "$CORR_OUT"
timeout --kill-after=60s 3600s python3 "$RUNNER_SAFE" \
  --excel "$CASE_XLSX" --sheet "$SHEET" \
  --start_line 2 --end_line 2 --run_mode 0 \
  --output_dir "$CORR_OUT" >"$CORR_OUT/run.log" 2>&1
```

主进程 exit 0 不能作为 PASS。必须检查 8 个 rank 的 y 和 expert-token CSV：

```bash
python3 - "$CORR_OUT" <<'PY'
import csv, sys
from pathlib import Path

root = Path(sys.argv[1])
log = (root / "run.log").read_text(errors="replace")
bad = ["[ERROR]", "Case FAILED", "Traceback", "continue test Failed"]
assert not any(x in log for x in bad), [x for x in bad if x in log]
assert log.count("continue test Success") == 8

with (root / "csv_result/mega_moe_result.csv").open(newline="") as f:
    y = [r for r in csv.reader(f) if len(r) >= 3 and r[1].startswith("output_x_rank")]
assert len(y) == 8 and {r[1] for r in y} == {f"output_x_rank{i}" for i in range(8)}
assert all(r[2] == "PASS" for r in y), y

with (root / "csv_result/moe_distribute.csv").open(newline="") as f:
    e = [r for r in csv.reader(f) if len(r) >= 3 and r[1].startswith("expert_token_nums_")]
assert len(e) == 8 and {r[1] for r in e} == {f"expert_token_nums_{i}" for i in range(8)}
assert all("PASS" in r[2] and "FAIL" not in r[2] for r in e), e
print("PASS: y and expert_token_nums on ranks 0..7")
PY
```

额外判定：日志中不得有 scalar GM invalid、hang、ready 永远不到；连续运行至少 10 次。
除固定 case 外，再补 M=0、尾 M、最后 expert/最后 wave 的定向 case；当前非法 tail-N
输入应由 tiling 拒绝，只有未来放宽 H 约束后才补 tail-N 数值 case。

最小地址 smoke 用单独包：

```bash
(cd "$TOKEN_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_GM_READY_SMOKE_TEST=1' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)
```

该模式提前返回，y/expert-token 必然不能过标准 correctness；只判定 8 rank 在 timeout 内
结束且没有 scalar GM 地址错误，然后恢复正式包。

## A/B 性能 `[NEEDS-NPU]`

四个正式候选：

| Variant | Branch | build macro |
|---|---|---|
| A token/per-expert | token | `READY_MODE=0` |
| B token/per-batch | token | 默认或 `READY_MODE=1` |
| C tile/per-expert | tile | 默认或 `READY_MODE=0` |
| D tile/per-batch | tile | `READY_MODE=1` |

对应的完整 build/install 命令如下；每次 build 后都按前文方法核验生成配置中的宏：

```bash
# A
(cd "$TOKEN_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_READY_MODE=0' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)

# B
(cd "$TOKEN_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_READY_MODE=1' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)

# C
(cd "$TILE_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_READY_MODE=0' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)

# D
(cd "$TILE_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_READY_MODE=1' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)

```

每组先跑 correctness。性能行是 Excel row 3，一次采 100 次；建议按
`A-B-B-A-A-B` 交错，token/tile 也使用相同交错方式。每次 trial 重新安装对应包并用独立
目录：

```bash
VARIANT=token_per_expert
TRIAL=01
PERF_OUT=$RUN_ROOT/$VARIANT/perf_trial_$TRIAL
mkdir -p "$PERF_OUT"
timeout --kill-after=60s 7200s python3 "$RUNNER_SAFE" \
  --excel "$CASE_XLSX" --sheet "$SHEET" \
  --start_line 3 --end_line 3 --run_mode 0 \
  --output_dir "$PERF_OUT" >"$PERF_OUT/run.log" 2>&1

python3 batch_run_once/export_batch_performance.py \
  --profiling-root "$PERF_OUT/Profiling_Result_mc2_cascade_16p" \
  --result-dir "$PERF_OUT/csv_result" \
  --result-csv "$PERF_OUT/csv_result/mega_moe_result.csv" \
  --output "$PERF_OUT/csv_result/batch_performance_result.xlsx" \
  --tail-size 50 --rank-per-dev 8 --server-index 0 \
  --tp-world-size 1 --shared-expert-rank-num 0
```

性能行也会做 correctness 比较，必须先用上一节严格 checker 验收。主指标为导出表中
`MegaMoe/moe` 的 `median_of_medians_us`，同时报告最慢 rank 的
`max_of_medians_us` 和 `profile_count`（应为 8）。不要嵌套 torch profiler 与外层
`msprof`；需要 msprof 时用 correctness row 单独采集。

空白观测表：

| Variant/trial | y+counts | no hang/scalar error | median-of-medians us | max-of-medians us | FIX/Atomic count | notes |
|---|---|---|---:|---:|---:|---|
| upstream/master |  |  |  |  |  |  |
| A1 token/expert |  |  |  |  | 576/rank expected |  |
| B1 token/batch |  |  |  |  | 144/rank expected |  |
| C1 tile/expert |  |  |  |  | 576/rank expected |  |
| D1 tile/batch |  |  |  |  | 144/rank expected |  |

## 内部 timeline `[NEEDS-NPU]`

构建专用诊断包；它默认关闭，打开时把 `expert_token_nums_out[16]` 改作 trace，因此不能
用于 correctness/performance：

```bash
(cd "$TOKEN_SRC" && \
  bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j16 -O3 \
    --ops-compile-options '-DMEGA_MOE_GMM2_WAVE_TIMELINE_DIAG=1' && \
  ./build_out/cann-ops-transformer-custom_linux-x86_64.run)
```

在安全 runner 的 `expert_token_nums = expert_token_nums.cpu()` 后临时加入：

```python
if os.getenv("MEGA_MOE_WAVE_TRACE") == "1":
    words = [int(x) & 0xffffffff for x in expert_token_nums.flatten().tolist()]
    print(f"WAVE_TRACE rank={rank} words={words}", flush=True)
```

设置 `MEGA_MOE_WAVE_TRACE=1` 跑 correctness row。y 仍应 PASS，但 expert-token compare
预期 FAIL；只在该专用 run 中忽略 count FAIL。每 rank 的 16 words 定义为：

| slot | marker |
|---:|---|
| 0,1 | AIV1 logical-0 Dispatch(B+1) begin/end |
| 2,3 | AIC0 首个有效 GMM1(B+1) dispatch wave ready / 整 wave control end |
| 4,5 | AIC0 GMM2(B) control-entry / end-after-ready-publish |
| 6,7 | AIV0 logical-0 SwiGLU(B+1) control-entry/end |
| 8,9 | AIV1 logical-0 Combine(B) control-entry/end |
| 10 | magic `0x57415645` (`WAVE`) |
| 11 | format version，当前 1 |
| 12,13,14,15 | blockNum, expertsPerWave, worldSize, readyMode |

cycle 只保留低 32 位，按 unsigned modular subtraction 解码。该轻量 marker 只记录
logical AIV1-0/AIV0-0/AIC0，能验证代表核的局部顺序：

```text
Dispatch(B+1) end <= Combine(B) control entry
```

slot 2 在 AIC0 复核首个有效 expert 的 dispatch-wave flag 达到目标行数后记录，随后才调用
GMM1，因此直接检查了该 GMM1 输入已经 ready；它不必晚于整个 B+1 所有 expert 的
prefetch 结束。再检查代表核的 `[6,7]` AIV0 SwiGLU 与 `[8,9]` AIV1 combine、
`[2,3]` AIC GMM1 与 combine 是否存在交叠。slot 8 是进入 combine 控制段的时间；
per-expert 的具体 GM read 仍在各 expert ready wait 之后。

代码保证的是每颗 AIV1 自己完整执行 `Dispatch(B+1)` 后才进入自己的 `Combine(B)`，
没有引入 prompt 所禁止的 batch-wide `SyncAll`。不同 AIV1 允许异步跨 wave；因此这个
core-0 trace 不能证明 group 级 `max(dispatch end) <= min(combine begin)`。是否要求并满足
aggregate 图语义必须在 NPU 上采 all-core task/profile 后验收，同时确认快核提前进入
combine/下一 wave 不会拖慢慢核 dispatch。还要确认 AIV0 没有 combine task，并记录
AIC/AIV0/AIV1 scalar、GM load 次数和端到端时间。

## BlockMmad Phase 7 API 边界

当前没有在尚未上板的 A-D 基础上叠加“单实例”开关。最新依赖改变了旧结论，但仍没有
足够的公开生命周期契约把该实验混入本次基础分支：

- 当前 generic 路径在
  `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:526` 的
  `GroupMatmulExecGeneric` 中每 expert 构造 `BlockMmad` 并 `Init`。
- QBMM MX `BlockMmad` 的构造/析构在
  `block_mmad_qbmm_mx.h:121-145` 成对设置/等待 event。ops-tensor `051664f3` 新增了
  `Init:156-159`，会把 `abL1LoopCnt_`、`scaleLoopCnt_`、`l0PingPong_` 和
  `l0cPingPong_` 全部清零；所以旧的“重复 Init 必然继承 counters”判断已经失效，
  “单对象生命周期 + 每 expert Init”现在具备进入独立编译实验的条件。
- 但普通类仍没有 `Reset/ReInit/UpdateParamsForNextProblem` 或公开的 expert-boundary
  drain/rearm API。把对象外提会同时删掉每 expert 析构中的 event wait，而 `Init` 只重置
  参数/phase，没有声明它是完整的 event/fixpipe rebind 契约；本地静态检查不能据此证明
  不 hang 或不错数。该候选必须是默认关闭的独立提交，并在 NPU 上先做 correctness/hang，
  不能直接并入 A-D 性能包。
- ops-tensor 另有 QGMM MX 专用 `BlockMmad`，具备
  `UpdateParamsForNextProblem`，其 `KernelGroupedMmadWithScaleMx` 也保持单 scheduler/
  BlockMmad 跨 group 连续双缓冲。但它依赖 grouped problem list、地址/layout 和专用
  scheduler，不是当前 bespoke GMM1/GMM2 调用的 drop-in replacement。对应入口在
  `block_mmad_qgmm_mx.h:124,152` 和 `kernel_qgmm_mx.h:183,377,451`。

因此先让 A-D 在 NPU 通过并选择实测组合；随后把 E 拆成“普通单对象 + 每 expert Init”
的独立宏实验，把 F 拆成 QGMM grouped contract 适配。两者都要单独验证 rolling cursor、
不同 M/地址、event/fixpipe、GMM1 post 和 GMM2 ready 边界。当前交付不声称已有 E/F
正确候选。

## 用户回传清单

请回传每个候选的：commit、完整宏、run 包 SHA256、8 rank correctness CSV、是否 hang/
scalar error、内部 trace、profiling xlsx，以及交错 trial 原始目录。后续再基于数据回答
per-batch 是否损失 expert 流式性、token/tile 哪个更快、AIV0 overlap 是否恢复，以及与
upstream/master 的剩余差距。
