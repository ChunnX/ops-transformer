# SlidingTileAttention 自验证说明

## 支持域与证据集

算子契约支持 rank-4 BNSD FP16/BF16；非空计算要求 `B > 0`、`1 <= N <= 128`、任意正的 8 倍数 `D`，
window 为长度 3 的广播三元组或长度 `3*N` 的逐 head 三元组，每个三元组的三个值均须为正奇数。
`q/k/v` 可为非连续 Tensor，由 ACLNN `Contiguous` materialize；shape 一致的任意空 Tensor 在
GetWorkspaceSize 快返且不进入 L0。

reviewer runner 共包含 22 个设备 case：前 18 个是历史固定形状 `N=8,D=64` 回归保护集，后 4 个为 case
19、23、24、25，覆盖动态 N/D/window、`B=2/3`、BF16 padded-D 和非连续 layout；另有一个空 Tensor API
检查。case 20、21、22、26 按本次 PR 自测范围不进入 runner，未在该 runner 中作性能结论；它们仍在算子
运行时 metadata 契约内，且不会触发 case-id 分派或 fallback。22 个执行 case 是代表性证据，不是完整支持
边界。所有设备 case 使用确定性、非零且非均匀的 Q/K/V，host 期望值显式应用 runtime
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
SLIDING_TILE_ATTENTION_WARMUP=3
SLIDING_TILE_ATTENTION_MEASURE_REPEAT=5
SLIDING_TILE_ATTENTION_REPORT_PATH=/absolute/path/sta_self_validation.csv
```

脚本构建当前 PR 源码、在 `/tmp` 下临时安装 `sta_review` vendor、编译 PR 内 C++ runner，并依次执行
22 个设备 case 和空 Tensor API 检查。退出时删除隔离安装目录，不修改活动 CANN 的 vendor 或
`vendors/config.ini`。成功标志为：

```text
API_PASS empty_tensor workspace=0
CASE_PASS id=... max_abs=... mae=... rmse=... tol=... latency_us_median=... warmup=3 repeat=5
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=22 report=... warmup=3 repeat=5
```

## 判定方法

runner 在完整长序列上执行当前 ACLNN custom op，使用独立的非零、非均匀 Q/K/V 和解析期望值检查
runtime scale、window、text、dtype 和分块输出写回；不读取 OpForge task/cases/golden 文件，不调用
framework/reference fallback。每个 case 对全部预定义采样点统计最大绝对误差、平均绝对误差和 RMSE；
FP16 最大绝对误差阈值为 `3e-3`，BF16 为 `2e-2`，任一采样点非有限或超过该 dtype 阈值即失败。

性能计时使用同一 ACLNN executor 与同一 stream 上的 ACL runtime event，不包含输入初始化、golden 计算、
输出采样或内存分配。每个 case 固定预热 3 次、测量 5 次，输出最小/中位/最大设备时间（微秒）。runner
默认在仓库根目录写入 `sliding_tile_attention_self_validation.csv`；CSV 每行包含 shape、dtype、采样数、
误差统计、阈值、计时统计、预热/重复次数和 layout。提交审查时应上传该 CSV 及完整 stdout 日志，二者均
带有 source-tree SHA256 和明确的 pass marker。

迁移来源的冻结 Ascend C 源码曾在 Ascend910_9382（physical=6, card=3, chip=0, logical=npu:0）通过
OpForge `eval_20260719_205421` 的 18/18 correctness gate，active-window geomean 为 17.4551x，最小为
10.2001x，18 个 case 均为单 kernel、零 device gap、零 active regression。该记录只覆盖上表固定形状，
不是当前动态实现的替代证据，也不证明非连续输入或空 Tensor 的 NPU 行为。

## 动态泛化验证状态

当前 reviewer runner 将 case 19、23、24、25 的动态 N/D/window、非连续输入 materialize 和空 Tensor
快返纳入可复跑入口。发布结论只引用当前 PR head fresh build 的原始日志；迁移来源的旧 18-case 结果仅
作为固定快路径历史保护证据，不替代当前源码验证。case 20、21、22、26 未进入本次 runner，不能由这份
22-case 日志推导其性能结果。

## BF16 与无 fallback 边界

BF16 路径满足一次外部 STA kernel launch。Q/K 保持 BF16，V tile 因当前 CATLASS PV lowering 限制在
同一 kernel 内 staging 为 half，并沿标准 online-softmax/PV/rescale 数据流直接写回 BF16；生产路径不
启用旧的双概率修正。该边界不增加独立 Cast kernel，也不使用 framework、CPU、vendor whole-op 或
其他后端 fallback。

`N=8,D=64` 固定快路径和其他合法 N/D 的动态通用路径都来自当前目录源码；非法 metadata 明确报错。
