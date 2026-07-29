# SlidingTileAttention 自验证说明

## 支持域与证据集

算子契约支持 rank-4 BNSD FP16/BF16；非空计算要求 `B > 0`、`1 <= N <= 128`、任意正的 8 倍数 `D`，
window 为长度 3 的广播三元组或长度 `3*N` 的逐 head 三元组，每个三元组的三个值均须为正奇数。
`q/k/v` 可为非连续 Tensor，由 ACLNN `Contiguous` materialize；shape 一致的任意空 Tensor 在
GetWorkspaceSize 快返且不进入 L0。

reviewer runner 共包含 22 个设备 case：前 18 个是历史固定形状 `N=8,D=64` 回归保护集，后 4 个为 case
19、23、24、25，覆盖动态 N/D/window、`B=2/3`、BF16 padded-D 和非连续 layout；另有一个空 Tensor API
检查。runner 固定执行下表 22 个 case，未列入表中的 workload 不由这份日志给出性能结论。22 个执行 case
是代表性证据，不是完整支持边界。所有设备 case 使用确定性、非零且非均匀的 Q/K/V，host 期望值显式应用 runtime
`1/sqrt(D)`；输出采样覆盖所有合法的 `0,7,63,64,127,128,D-1` 列。tile=`[6,8,8]`。

| case | dtype | shape `[B,N,S,D]` | text | seqShape | window | 覆盖点 |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | source mix | FastVideo 正式有 text |
| 2 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | source mix | BF16 有 text |
| 3 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | wide mix | 边界 clamp |
| 4 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | wide mix | BF16 clamp |
| 5 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | source mix | no-text source mix |
| 6 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | source mix | BF16 no-text |
| 7 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | wide mix | wide clamp |
| 8 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | wide mix | BF16 wide clamp |
| 9 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | uniform 333 | 正式性能形状 |
| 10 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | uniform 333 | BF16 性能形状 |
| 11 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | uniform 333 | text + uniform |
| 12 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | uniform 333 | BF16 text + uniform |
| 13 | FP16 | `[1,8,69120,64]` | 0 | 18x48x80 | uniform 333 | Wan 画布 |
| 14 | BF16 | `[1,8,69120,64]` | 0 | 18x48x80 | uniform 333 | BF16 Wan 画布 |
| 15 | FP16 | `[1,8,69120,64]` | 0 | 18x48x80 | Wan mixed | odd window mix |
| 16 | BF16 | `[1,8,69120,64]` | 0 | 18x48x80 | Wan mixed | BF16 odd mix |
| 17 | FP16 | `[2,8,69120,64]` | 0 | 18x48x80 | uniform 333 | batch=2 保护 |
| 18 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | 单 `[3,3,3]` | window 广播保护 |
| 19 | FP16 | `[1,3,69120,72]` | 0 | 18x48x80 | 逐 head `3*N` | 动态 N/D 与逐 head window |
| 23 | FP16 | `[1,1,69120,72]` | 0 | 18x48x80 | 单 `[3,3,3]` | 非连续 D-stride input/output |
| 24 | FP16 | `[3,1,69120,72]` | 0 | 18x48x80 | 单 `[2^32+1,2^32+1,2^32+1]` | `B=3` 与动态 D；检查最后 batch 和宽 INT64 window 的全轴饱和 |
| 25 | BF16 | `[2,3,69120,72]` | 0 | 18x48x80 | 逐 head `3*N`，T/H/W 分别含 `2^32+1` | BF16 padded-D、逐 head window、三轴 INT64 与 D72 尾交叉覆盖 |

## Host 契约测试

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
    experimental/attention/sliding_tile_attention/tests/case_guard_test.cpp \
    -Iexperimental/attention/sliding_tile_attention/op_host \
    -o /tmp/sta_case_guard_test
/tmp/sta_case_guard_test
```

预期标志：

```text
ALL_SLIDING_TILE_ATTENTION_CASE_GUARD_TESTS_PASSED dynamic_heads=1..128 head_dim_multiple=8
```

该测试验证 metadata 规则和 window 规范化，不替代 NPU kernel correctness。

## 22-case Reviewer 复跑

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_22cases.sh
```

可通过环境变量覆盖默认值：

```text
SLIDING_TILE_ATTENTION_SOC_VERSION=ascend910_93
SLIDING_TILE_ATTENTION_DEVICE_ID=0
SLIDING_TILE_ATTENTION_BUILD_THREADS=16
SLIDING_TILE_ATTENTION_PYTHON=/path/to/python3
SLIDING_TILE_ATTENTION_WARMUP=3
SLIDING_TILE_ATTENTION_MEASURE_REPEAT=5
SLIDING_TILE_ATTENTION_REPORT_PATH=/absolute/path/sta_self_validation.csv
```

脚本要求选定的 Python 可导入 `packaging`，并将其用于 ops 构建；默认使用 `PATH` 中的 `python3`，可通过
`SLIDING_TILE_ATTENTION_PYTHON` 显式指定。脚本构建当前 PR 源码、在 `/tmp` 下临时安装 `sta_review`
vendor、编译 PR 内 C++ runner，并依次执行
22 个设备 case 和空 Tensor API 检查。退出时删除隔离安装目录，不修改活动 CANN 的 vendor 或
`vendors/config.ini`。成功标志为：

```text
API_PASS empty_tensor workspace=0
CASE_PASS id=... max_abs=... mae=... rmse=... tol=... latency_us_median=... warmup=3 repeat=5
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=22 report=... warmup=3 repeat=5
```

## 本次 fresh 结果

2026-07-30 04:31--04:32 在
`Ascend910_9382 (physical=9, card=4, chip=1, logical=npu:0 after visibility remap)` 上，使用
`ascend910_93` package 路径完成了一次当前源码的完整构建、隔离安装和 reviewer runner 复跑。执行范围为
case `1..19,23..25`，结果为 22/22 通过；空 Tensor 检查返回 `workspace=0`。原始证据如下：

- [设备执行结果日志](evidence/run_22cases.log)，由同一次完整构建日志中原样保留源码摘要、22 条
  `CASE_PASS`、空 Tensor 和最终通过标志，SHA256
  `dc6b571836b5a359cf2e2be518d5ff203a0b48e454d7754f88fe090ed171b47c`。提交版本在原始 stdout 前补充
  运行开始时间和设备身份，并清理第三方构建输出的行尾空白；未改动源码摘要、case 行、数值或通过标志。
- [逐 case CSV](evidence/self_validation.csv)，SHA256
  `395ba65558fdc63015831306ea3f288d6611125ce669049dc05e8f6d577bbfc1`。
- build 前脚本记录的源码树摘要为
  `b776888815e70cc63a95290aced1af8647f0c6d2acd0f67d3cea1f6416467c14`。
- [源码、复现器和证据 SHA256 清单](evidence/SHA256SUMS) 可用
  `sha256sum -c experimental/attention/sliding_tile_attention/docs/evidence/SHA256SUMS` 直接校验。

源码树摘要覆盖当前交付目录中的构建、实现、runner 和测试源码，并额外覆盖仓库级 CATLASS 复制逻辑
`cmake/func.cmake`；不包含文档与 `docs/evidence/`，因此文档补充和证据刷新不会形成自引用漂移。日志末尾的
判定标志为：

```text
API_PASS empty_tensor workspace=0
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=22 ... warmup=3 repeat=5
```

CSV 中的 `latency_us_*` 是同一设备上 ACL runtime event 记录的二阶段 executor 执行时间，不含一阶段
GetWorkspaceSize，也没有配对 baseline。它们用于复跑稳定性和异常定位，不构成 speedup 或官方性能通过。

candidate 为当前 PR 构建的 `aclnnSlidingTileAttention`；reference 为 runner 内独立解析实现。output shape
与 input shape 相同，`median us` 为 5 次设备测量中位数。完整 min/max、MAE 和 RMSE 见原始 CSV。

| case | dtype | input/output shape | elements | layout | candidate / reference | max abs / threshold | median us | coverage |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | --- |
| 1 | FP16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000104398 / 0.003 | 135737.953 | text, source window |
| 2 | BF16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000946879 / 0.02 | 71183.680 | BF16 text |
| 3 | FP16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000104845 / 0.003 | 241117.031 | wide window clamp |
| 4 | BF16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000946879 / 0.02 | 123040.023 | BF16 wide clamp |
| 5 | FP16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000115752 / 0.003 | 101958.742 | no-text source window |
| 6 | BF16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000760108 / 0.02 | 48192.820 | BF16 no-text |
| 7 | FP16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000115752 / 0.003 | 189739.250 | no-text wide clamp |
| 8 | BF16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000949711 / 0.02 | 86137.164 | BF16 wide clamp |
| 9 | FP16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000105858 / 0.003 | 26891.221 | broadcast-equivalent window |
| 10 | BF16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000811160 / 0.02 | 17301.900 | BF16 broadcast-equivalent |
| 11 | FP16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000096917 / 0.003 | 37529.098 | text and uniform window |
| 12 | BF16 | `[1,8,115328,64]` | 59047936 | contiguous | PR ACLNN / analytic host | 0.000917137 / 0.02 | 24525.379 | BF16 text and uniform window |
| 13 | FP16 | `[1,8,69120,64]` | 35389440 | contiguous | PR ACLNN / analytic host | 0.000109047 / 0.003 | 22391.000 | Wan canvas |
| 14 | BF16 | `[1,8,69120,64]` | 35389440 | contiguous | PR ACLNN / analytic host | 0.000205591 / 0.02 | 14434.820 | BF16 Wan canvas |
| 15 | FP16 | `[1,8,69120,64]` | 35389440 | contiguous | PR ACLNN / analytic host | 0.000083119 / 0.003 | 54810.781 | mixed odd window |
| 16 | BF16 | `[1,8,69120,64]` | 35389440 | contiguous | PR ACLNN / analytic host | 0.000952005 / 0.02 | 28932.480 | BF16 mixed odd window |
| 17 | FP16 | `[2,8,69120,64]` | 70778880 | contiguous | PR ACLNN / analytic host | 0.000109047 / 0.003 | 44479.098 | batch 2 |
| 18 | BF16 | `[1,8,82944,64]` | 42467328 | contiguous | PR ACLNN / analytic host | 0.000811160 / 0.02 | 17317.779 | broadcast window API |
| 19 | FP16 | `[1,3,69120,72]` | 14929920 | contiguous | PR ACLNN / analytic host | 0.000078529 / 0.003 | 21174.879 | dynamic N/D, per-head window |
| 23 | FP16 | `[1,1,69120,72]` | 4976640 | non-contiguous D stride | PR ACLNN / analytic host | 0.000115693 / 0.003 | 6214.760 | input/output materialize |
| 24 | FP16 | `[3,1,69120,72]` | 14929920 | contiguous | PR ACLNN / analytic host | 0.000091955 / 0.003 | 137445.641 | batch 3, wide INT64 window |
| 25 | BF16 | `[2,3,69120,72]` | 29859840 | non-contiguous D stride | PR ACLNN / analytic host | 0.000324398 / 0.02 | 183695.688 | BF16 padded D, per-head window |

## 完整 public_v3 补充证据

与当前交付实现对应的独立 OpForge 完整评测为 `eval_20260729_185640`：当前 `public_v3` 26/26
精度通过、0 failed，`active_vs_active` 几何平均为 `10.572285683x`，25/26 case 高于 `1x`。评测状态仍为
`PERF_REGRESSION`，唯一 active-window 回归是 case 26：candidate `40140.36 us`，Torch baseline
`35873.70 us`，即 `0.893706484x`。因此该结果只支持“完整公开集精度通过”，不支持“完整公开集性能通过”。

同一 evaluator snapshot 的固定种子 generalization audit 为 40/40 通过，integrity report 的 error-severity
finding 为 0。该完整评测与本目录 22-case reviewer runner 是两套独立证据；`run_22cases.sh` 不读取或重放
OpForge task/cases/golden，也不据此生成本页的源仓自验证通过标志。

## 判定方法

runner 在完整长序列上执行当前 ACLNN custom op，使用独立的非零、非均匀 Q/K/V 和解析期望值检查
runtime scale、window、text、dtype 和分块输出写回；不读取 OpForge task/cases/golden 文件，不调用
framework/reference fallback。每个 case 对全部预定义采样点统计最大绝对误差、平均绝对误差和 RMSE；
FP16 最大绝对误差阈值为 `3e-3`，BF16 为 `2e-2`，任一采样点非有限或超过该 dtype 阈值即失败。

性能计时使用同一 stream 上的 ACL runtime event，不包含一阶段 GetWorkspaceSize、输入初始化、golden 计算、
输出采样或内存分配。ACLNN 二阶段执行会自动释放 executor，因此每次预热和测量都重新调用一阶段创建
独立 executor，workspace 仅按返回容量复用；不会重复执行已经消费的 executor。每个 case 固定预热 3 次、
测量 5 次，输出最小/中位/最大设备时间（微秒）。runner 默认在仓库根目录写入
`sliding_tile_attention_self_validation.csv`；CSV 每行包含 shape、dtype、采样数、误差统计、阈值、计时
统计、预热/重复次数和 layout。完整 stdout 日志带有 source-tree SHA256 和明确的 pass marker；CSV 由
`docs/evidence/SHA256SUMS` 及同目录证据清单认证。

## 动态泛化验证状态

当前 reviewer runner 将 case 19、23、24、25 的动态 N/D/window、非连续输入 materialize 和空 Tensor
快返纳入可复跑入口。发布结论只引用当前 PR head fresh build 的原始日志；未列入上表的 workload 不由这份
22-case 日志推导其精度或性能结果。

## BF16 与无 fallback 边界

BF16 路径满足一次外部 STA kernel launch。`D=64` 由 tiling key 1 选择 solution-owned lean 编译变体；
其他合法 BF16 D 由 key 0 通用变体执行。Q/K 保持 BF16，V tile 因当前 CATLASS PV lowering 限制在同一
kernel 内 staging 为 half，并沿标准 online-softmax/PV/rescale 数据流直接写回 BF16；生产路径不启用旧的
双概率修正。该边界不增加独立 Cast kernel，也不使用 framework、CPU、vendor whole-op 或其他后端
fallback。key 0/1 和其内部固定/dynamic runtime 路由都来自当前目录源码；非法 metadata 明确报错。
